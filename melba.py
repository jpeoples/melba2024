"""This module contains all code to reproduce the analysis in our 2024 MELBA paper

"XXX"

TODO.
"""

import fnmatch
import json
import os
import sys

from matplotlib import pyplot
pyplot.rcParams.update({'font.size': 14})

import mrmr

import numpy
import pandas


from scipy.stats import wilcoxon
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from entrypoints import EntryPoints

entry = EntryPoints()

class Utils:
    def __init__(self, conf):
        self.conf = conf

    @classmethod
    def from_file(cls, fname):
        with open(fname) as f:
            obj = json.load(f)
            return cls(obj)

    def r_run(self, script, args):
        rpath = self.conf['r_path']
        scriptpath = self.conf['r_scripts'][script]
        arg_string = " ".join(args)

        os.system(f'"{rpath}" {scriptpath} {arg_string}')

    def data_path(self, path, prefix=None, external=False, write=False):
        if external:
            fullpath=external
        else:
            parts = [self.conf['data']]
            if prefix:
                parts.append(prefix)
            parts.append(path)
            fullpath = os.path.join(*parts)

        if write:
            self.ensure_dir(fullpath)
                
        return fullpath

    @staticmethod
    def ensure_dir(fpath):
        dir = os.path.dirname(fpath)
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        return fpath

    def input_path(self, path):
        return self.data_path(path, prefix='input')

    def working_path(self, path, write=False):
        return self.data_path(path, prefix="working", write=write)

    def output_path(self, path, write=False):
        return self.data_path(path, prefix="output", write=write)

    def load_repro_tables(self):
        tables = {k: load_repro_table(self.data_path(p)) for k, p in self.conf['repro_tables'].items()}
        cat = []
        for ext, tab in tables.items():
            tab['extractor'] = ext
            tab = tab.set_index('extractor', append=True)
            tab = tab.reorder_levels(('extractor', 'patient', 'timepoint', 'spacing', 'asir'))
            cat.append(tab)

        return pandas.concat(cat, axis=0)

    def load_survival_table(self, feature_set):
        assert feature_set in self.conf['survival_tables']
        p = self.data_path(self.conf['survival_tables'][feature_set])
        return load_survival_table(p)


    def load_survival_tables(self):
        tables = {k: load_survival_table(self.data_path(p)) for k, p in self.conf['survival_tables'].items()}
        cat = []
        for ext, tab in tables.items():
            tab.index.name = 'patient'
            tab['extractor'] = ext
            tab = tab.set_index('extractor', append=True)
            tab = tab.reorder_levels(('extractor', 'patient'))
            tab = value_filter(tab) # Drop cases that are largely constant
            cat.append(tab)

        return pandas.concat(cat, axis=0)

    def load_survival_clin_table(self):
        table = pandas.read_csv(self.data_path(self.conf['survival_data']), index_col=0)
        return table

    def univariate_survival_outcome(self):
        tab = self.load_survival_clin_table()
        conf = self.conf['univariate_survival']
        time = conf['time']
        event = conf['event']

        return pandas.DataFrame({"time": tab[time], "event": tab[event]})

    def multivariate_survival_outcome(self):
        tab = self.load_survival_clin_table()
        conf = self.conf['multivariate_survival']
        time = conf['time']
        event = conf['event']

        return pandas.DataFrame({"time": tab[time], "event": tab[event]})

    # Deterministic function to get the nth seed
    def get_seed(self, repeat):
        seed = self.conf['multivariate_survival']['seed_generator_seed']
        rng = numpy.random.RandomState(seed)
        return rng.randint(2**31, size=(repeat+1,))[-1]

    def get_random_state(self, repeat):
        seed = self.get_seed(repeat)
        return numpy.random.RandomState(seed)

    def multivariate_survival_features(self, args):
        fs = self.conf['multivariate_survival']['feature_sets'][args.feature_set]
        if fs == 'all':
            features = self.load_survival_tables()
            features = blast_out_extractors(features)
        else:
            features = self.load_survival_table(fs)
        
        return features

    def multivariate_feature_selector(self, args):
        def cccs_for_feature_set(args):
            fs = self.conf['multivariate_survival']['feature_sets'][args.feature_set]
            cccs = pandas.read_csv(self.working_path('lmm_cccs.csv'), index_col=[0,1])

            if fs == "all":
                cccs = cccs['ccc']
                cccs.index = cccs.index.map(lambda s: f"{s[0]}_{s[1]}")
            else:
                cccs = cccs.loc[fs, 'ccc']

            return cccs

        cccs = cccs_for_feature_set(args)
        thresh = self.conf['multivariate_survival']['ccc_threshold']
        thresh_keep = cccs >= thresh
        thresh_keep = cccs.index[thresh_keep]
        constructors = {
            "mRMR": lambda n: mRMRFeatSel(n, None),
            "mRMRReproThresh": lambda n: ReproFeatSel(n, thresh_keep),
            "mRMRReproWeighted": lambda n: mRMRFeatSel(n, cccs)
        }
        feature_selection = self.conf['multivariate_survival']['feature_selection'][args.feature_selection]

        return constructors[feature_selection]

    def multivariate_survival_model(self, args):
        nfeat = self.conf['multivariate_survival']['feature_counts'][args.feature_count]
        fsel_constructor = self.multivariate_feature_selector(args)
        return MultivariateSurvival(nfeat, fsel_constructor)

    def multivariate_cross_validation(self, args, repeat):
        nfolds = self.conf['multivariate_survival']['cv_n_fold']
        repeats = self.conf['multivariate_survival']['cv_repeats']
        assert repeat <= repeats

        seed = self.get_random_state(repeat)
        cv = CrossValidation(StratifiedKFold(nfolds, shuffle=True, random_state=seed))
        return cv

    def multivariate_combinations(self):
        conf = self.conf['multivariate_survival']

        repeats = list(range(conf['cv_repeats']))
        counts=conf['feature_counts']
        fsets = conf['feature_sets']
        fsels = conf['feature_selection']

        return set([(st, cnt, sel, rep) for st in fsets for cnt in counts for sel in fsels for rep in repeats])

        


def blast_out_extractors(tab):
    out = []
    for ext, tab in tab.groupby('extractor'):
        tab = tab.loc[ext]
        tab.columns = [f"{ext}_{c}" for c in tab.columns]
        out.append(tab)

    return pandas.concat(out, axis=1)

class CrossValidation:
    def __init__(self, cv):
        self.cv = cv

    def split(self, X, y, groups=None):
        return self.cv.split(X, y['event'], groups)
    

    def get_n_splits(self, X, y, groups=None):
        return self.cv.get_n_splits(X, y, groups)



class Normalizer:
    def compute_normalization(self, X):
        self.threshold_ = VarianceThreshold()
        X = self.threshold_.fit_transform(X)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X)

    def normalize(self, X):
        Xout = pandas.DataFrame(self.scaler_.transform(self.threshold_.transform(X)), index=X.index, columns=self.threshold_.get_feature_names_out(X.columns))
        return Xout


class mRMRFeatSel:
    def __init__(self, nfeat=10, feat_prefs=None):
        self.nfeat=nfeat
        self.feat_prefs = feat_prefs

    def select_features(self, X, y):
        selected = mrmr_surv(X, y, self.nfeat, relevance=lambda X, y: surv_rel(X, y, feature_prefs=self.feat_prefs), show_progress=False, n_jobs=1)
        self.selected_ = selected

    def get_selected(self, X):
        return X.loc[:, self.selected_]

def feature_agglom_by_thresh(features, thresh):
    from sklearn.cluster import FeatureAgglomeration
    #print("...clustering")
    corr = features.corr(method='spearman').abs()
    cl = FeatureAgglomeration(distance_threshold=1-thresh, n_clusters=None, metric='precomputed', linkage='single')
    cl.fit(1-corr)

    return pandas.Series({ix: cli for ix, cli in zip(features.columns, cl.labels_)})
def univar_cox(surv, ft):
    from lifelines.exceptions import ConvergenceError
    fitter = CoxPHFitter()
    data = surv.join(ft)
    data = data.dropna(axis='index')

    try:
        fitter.fit(data, duration_col="time", event_col="event")
        res = fitter.summary.loc[ft.name, ['exp(coef)', 'p']]
        res.index = ['HR', 'p-value']
    except ConvergenceError:
        res = pandas.Series({"HR": 1, "p-value": 1})
        print(f"Failed on {ft.name}")

    return res

def select_best_features_from_clusters(clusters, features, surv):
    clusters = clusters[features.columns.intersection(clusters.index)]
    #print("...ranking")
    #ranking = univar_rank(time_col, event_col, features)
    #print('...somers')
    somers = surv_rel(features, surv)
    #ranking['somers'] = somers
    #print(ranking[['p-value', 'somers']].corr(method='spearman'))
    #all_selected = ranking.loc[ranking['p-value'] < 0.05].index

    all_selected = []
    #print("...sorting through")
    for cli, cluster_features in clusters.groupby(clusters):
        cluster_ranks = somers[cluster_features.index]
        sorted = cluster_ranks.sort_values(ascending=False)
        for ix, d in sorted.items():
            if d < 0.1:
                break
            res = univar_cox(surv, features[ix])
            if res['p-value'] < 0.1:
                all_selected.append(ix)
                break
        #best = cluster_ranks.idxmax()
        #if cluster_ranks[best] > 0.1:
        #    all_selected.append(best)
        #cluster_ranks = ranking.loc[cluster_features.index, "p-value"]
        #selected = cluster_ranks.sort_values()
        #for ix, pval in selected.items():
        #    if pval < 0.1: #and cccs.loc[preproc, ix] > 0.9:
        #        all_selected.append(ix)
        #        break

    #print("...ordering")
    selected_ranks = somers.loc[all_selected]
    order = selected_ranks.sort_values().index


    return features.loc[:, order]
class HierarchicalFeatureSelector:
    def __init__(self, thresh):
        self.corr_thresh = thresh

    def select_features(self, X, y):
        clusters = feature_agglom_by_thresh(X, self.corr_thresh)
        selfeat = select_best_features_from_clusters(clusters, X, y)

        self.selected_ = selfeat.columns.tolist()
        print(len(self.selected_), clusters.max()+1)

        return self

    def get_selected(self, ds):
        return ds.loc[:, self.selected_]

class ReproFeatSel:
    def __init__(self, nfeat, repro_features):
        self.repro_features = repro_features
        self.nfeat = nfeat


    def select_features(self, X, y):
        X = X[self.repro_features]
        self.mrmr_ = mRMRFeatSel(self.nfeat)
        self.mrmr_.select_features(X, y)

    def get_selected(self, X):
        return self.mrmr_.get_selected(X)


def surv_rel(X, y, feature_prefs=None):
    ci = compute_harrels_per_features(X, y)
    dxy = 2 * ci - 1
    adxy = dxy.abs()
    oci = (adxy + 1) / 2

    if feature_prefs is not None:
        oci = (oci - oci.min()) / (oci.max() - oci.min())
        fp = (feature_prefs - feature_prefs.min()) / (feature_prefs.max() - feature_prefs.min())
        oci = oci * fp

    return oci



def mrmr_surv(
        X, y, K,
        relevance = None, redundancy='c', denominator='mean',
        cat_features=None, cat_encoding='leave_one_out',
        only_same_domain=False, return_scores=False,
        n_jobs=-1, show_progress=True
):
    rels = relevance(X, y)
    X = X.loc[:, rels > 0.55]
    to_keep = []
    for ft in X.columns:
        res = univar_cox(y, X[ft])
        if res['p-value'] < 0.1:
            to_keep.append(ft)

    X = X.loc[:, to_keep]
    print("Remaining to select", X.shape[1])

    if relevance is None:
        relevance = surv_rel
    return mrmr.mrmr_classif(X, y, K, relevance=relevance, redundancy=redundancy, denominator=denominator, cat_features=cat_features, cat_encoding=cat_encoding, only_same_domain=only_same_domain, return_scores=return_scores, n_jobs=n_jobs, show_progress=show_progress)


class MultivariateSurvival(BaseEstimator):
    def __init__(self, nfeat=10, feat_sel=mRMRFeatSel):
        self.nfeat = nfeat
        self.feat_sel = feat_sel

    def fit(self, X, y):
        self.normalizer_ = Normalizer()
        self.feature_selector_ = self.feat_sel(self.nfeat)

        self.normalizer_.compute_normalization(X)
        X = self.normalizer_.normalize(X)

        self.feature_selector_.select_features(X, y)
        X = self.feature_selector_.get_selected(X)
        penalizers = [0, 0.001, 0.01, 0.1]
        for penalizer in penalizers:
            self.cph_ = CoxPHFitter(penalizer=penalizer)
            try:
                self.cph_.fit(X.join(y), duration_col='time', event_col='event')
            except:
                success = False
            else:
                success = True

            if success:
                break
        else:
            raise RuntimeError("Could not solve")


    def predict(self, X):
        X = self.feature_selector_.get_selected(self.normalizer_.normalize(X))
        pred = -self.cph_.predict_partial_hazard(X)
        return pred

    def score_predictions(self, pred, y):
        pred = pred.loc[y.index]

        return concordance_index(y['time'], pred, y['event'])

    def score(self, X, y):
        assert X.index.equals(y.index)
        pred = self.predict(X)

        return concordance_index(y['time'], pred, y['event'])



def value_filter(features, count=10):
    def cntr(x):
        counts = x.value_counts()
        counts = counts / counts.sum()
        if counts.max() > 0.5:
            return 1
        else:
            return 0
    bad_mask = features.apply(cntr, axis=0)

    return features.loc[:, bad_mask==0]

def cmpxs(tab, a):
    return tab.xs(a[0], level='reference_thickness').xs(a[1], level='comparison_thickness')

    
def concordance_correlation_coefficient(y_true, y_pred):
    np = numpy
    cov = np.cov(y_true, y_pred, ddof=0)
    mean_true=np.mean(y_true)
    mean_pred=np.mean(y_pred)

    var_true = cov[0,0]
    var_pred = cov[1,1]

    covar = cov[0,1]
    return 2 * covar / (var_true + var_pred + (mean_true-mean_pred)**2)

def parse_repro_caseid(caseid):
    spl = caseid.split("_")
    asir = int(spl[-1])
    slice_thickness = int(spl[-2])/100
    timepoint = spl[-3]
    subject = "_".join(spl[:-3])

    return subject, timepoint, slice_thickness, asir

def expand_repro_index(features):
    findex = features.index
    findex = [parse_repro_caseid(id) for id in findex]
    findex = pandas.MultiIndex.from_tuples(findex)
    features.index = findex
    features.index.names = ["patient", "timepoint", "spacing", "asir"]
    return features

def drop_diagnostics(features):
    to_drop = fnmatch.filter(features.columns, "*diagnostic*")
    return features.drop(columns=to_drop)

def unnormalized_features():
    features_to_drop = ['original_firstorder_Energy', 'original_gldm_DependenceNonUniformity',
       'original_gldm_GrayLevelNonUniformity', 'original_ngtdm_Busyness',
       'original_ngtdm_Coarseness', 'original_ngtdm_Strength',
       'original_glrlm_GrayLevelNonUniformity',
       'original_glrlm_RunLengthNonUniformity',
       'original_glszm_GrayLevelNonUniformity',
       'original_glszm_SizeZoneNonUniformity']

    features_to_drop = [aft for ft in features_to_drop for aft in ["liver_"+ft, "tumor_"+ft]]
    return features_to_drop

def drop_unnormalized(tab):
    features_to_drop = unnormalized_features()
    return tab.drop(columns=features_to_drop)    

def load_repro_table(fname):
    tab = pandas.read_csv(fname, index_col=0)
    tab = drop_diagnostics(tab)
    tab = drop_unnormalized(tab)
    tab = expand_repro_index(tab)

    return tab

def load_survival_table(fname):
    tab = pandas.read_csv(fname, index_col=0)
    tab = drop_diagnostics(tab)
    tab = drop_unnormalized(tab)
    return tab


@entry.add_common_parser
def common_parser(parser):
    parser.add_argument("--conf", required=False, default="conf.json")

def compute_pairwise_ccs(a, b):
    cols = a.columns
    assert cols.equals(b.columns)
    results = {}
    for col in cols:
        x = a[col]
        y = b[col]
        ccc = concordance_correlation_coefficient(x,y)
        results[col] = ccc

    return pandas.Series(results)



@entry.point
def pairwise_cccs(args):
    def _pairwise_ccctab(tab, ref, comp):
        tabs = []
        for ext, table in tab.groupby('extractor'):
            table = table.loc[ext]
            reft = table.xs(ref, level='spacing')
            compt = table.xs(comp, level='spacing')
            cccs = compute_pairwise_ccs(reft, compt).to_frame('ccc')
            cccs.index.name = 'full_roi_feature'
            cccs['comparison_thickness'] = comp
            cccs['reference_thickness'] = ref
            cccs['extractor'] = ext
            cccs['roi'] = cccs.index.map(lambda s: 'liver' if s.startswith('liver') else 'tumor')
            cccs['feature_name'] = cccs.index.map(lambda s: s.replace('liver_', '').replace('tumor_', ''))

            tabs.append(cccs)
        
        return pandas.concat(tabs, axis=0).set_index(['reference_thickness', 'comparison_thickness', 'extractor'], append=True)

    def _diff_tab(ccc, rfca, rfcb):
        tabs = []
        for ext, tab in ccc.groupby('extractor'):
            a = tab.xs(rfca[0], level='reference_thickness').xs(rfca[1], level='comparison_thickness')
            b = tab.xs(rfcb[0], level='reference_thickness').xs(rfcb[1], level='comparison_thickness')
            diffs = (a['ccc'] - b['ccc']).to_frame('ccc_diff')
            for col in a.columns:
                if col=='ccc':
                    continue
                assert a[col].equals(b[col])
            
            diffs = diffs.join(a.drop(columns=['ccc']))
            diffs['compared_a'] = None
            diffs['compared_b'] = None
            diffs['extractor'] = ext
            diffs['compared_a'] = diffs['compared_a'].apply(lambda x: rfca)
            diffs['compared_b'] = diffs['compared_b'].apply(lambda x: rfcb)
            tabs.append(diffs.set_index(['compared_a', 'compared_b', 'extractor'], append=True))

        return pandas.concat(tabs, axis=0)
                


    util = Utils.from_file(args.conf)
    tables = util.load_repro_tables()

    ix = pandas.IndexSlice
    # Select ASiR 20, select timepoint=clinical (there are no others anyway)
    tables = tables.xs('clinical', level='timepoint').xs(20, level='asir')

    cccs = pandas.concat(
        (
            _pairwise_ccctab(tables, 5.0, 3.75),
            _pairwise_ccctab(tables, 5.0, 2.5),
            _pairwise_ccctab(tables, 3.75, 2.5),
        ), axis=0
    )
    diff_table = pandas.concat(
        (
            _diff_tab(cccs, (5.0,3.75), (5.0,2.5)),
            _diff_tab(cccs, (5.0,3.75), (3.75,2.5)),
            _diff_tab(cccs, (5.0,2.5), (3.75,2.5)),
        ), axis=0
    )
    ccc_table = cccs
    ccc_table.to_csv(util.working_path('pairwise_cccs_20.csv'))
    diff_table.to_csv(util.working_path('pairwise_ccc_diffs_20.csv'))


class FigSet:
    def __init__(self):
        self.figs = []
        self.wmultiple = 6.4
        self.hmultiple = 4.8
        self.dpi = 600

    def make_fig(self, w, h, nrow=1, ncol=1):
        plt = pyplot
        fig, ax = plt.subplots(nrow, ncol, figsize=(self.wmultiple*w, self.hmultiple*h), dpi=self.dpi, layout='constrained')
        self.figs.append(fig)
        return fig, ax

    def set_dpi(self, x):
        for fig in self.figs:
            fig.set_dpi(x)

def box_plot(ax, features, mean_sorting=True, ascending=False, **kwargs):
    if mean_sorting:
        mean_sorting = features.mean(axis='rows').sort_values(ascending=ascending).index
        features = features.loc[:, mean_sorting]

    ax.boxplot([features[c].dropna() for c in features], **kwargs)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(numpy.arange(0, 1.1, 0.2))
    ax.set_ylabel("CCC")

    return ax, features

def compute_pval_celltext(cccs, xorder):
    cmps = [
        [(5.0, 3.75), (3.75, 2.5)],
        [(5.0, 3.75), (5.0, 2.5)],
        [(3.75, 2.5), (5.0, 2.5)]
    ]

    pvals = {}
    for ext, tab in cccs.groupby('extractor'):
        tab = tab.xs(ext, level='extractor')
        for (a,b) in cmps:
            at = cmpxs(tab, a)
            bt = cmpxs(tab, b)
            p = wilcoxon(at['ccc'], bt['ccc'], alternative='greater', nan_policy='omit').pvalue
            pvals.setdefault(ext, {})[(a,b)] = p

    cellvals_table = pandas.DataFrame.from_dict(pvals, orient='index').transpose()[xorder]
    cell_text = [list(x) for x in cellvals_table.applymap(lambda n: f"{n:0.1e}").to_numpy()]
    return cell_text

def pairwise_fixed_asir_plot(util, cccs, figset):
    fig, ax = figset.make_fig(2,1)
    hue = cccs.index.map(lambda x: "{} vs. {}".format(*x[1:]))
    hue.name = 'Comparison'
    hue_order = ['5.0 vs. 3.75', '3.75 vs. 2.5', '5.0 vs. 2.5']
    xorder = list(reversed(cccs.xs(5.0, level='reference_thickness').xs(3.75, level='comparison_thickness').groupby('extractor')['ccc'].mean().sort_values().index))
    print(xorder)
    sns.boxplot(cccs, x='extractor', y='ccc', hue=hue, order=xorder, hue_order=hue_order)
    ax.legend(loc='lower left')
    ax.set_xticks([])
    ax.set_xlabel(None)
    ax.set_ylabel("CCC")

    rowlabels = ["       vs.       ", "       vs.       ", "       vs.       "]
    cell_text = compute_pval_celltext(cccs, xorder)

    table =  ax.table(cellText=cell_text, colLabels=xorder, rowLabels=rowlabels, loc='bottom')
    table.scale(1,1.5)

    fig.savefig(util.output_path("figs/pairwise_cccs_boxplot.pdf", write=True))

@entry.point
def fig_pairwise_cccs(args):
    util = Utils.from_file(args.conf)
    cccs = pandas.read_csv(util.working_path('pairwise_cccs_20.csv'), index_col=[0,1,2,3])

    import seaborn as sns

    figs = FigSet()

    pairwise_fixed_asir_plot(util, cccs, figs)
    
    figs.set_dpi(72)
    pyplot.show()

@entry.point
def lmm_cccs(args):
    util = Utils.from_file(args.conf)

    features = util.load_repro_tables()
    lmm_results = []
    for ext, tab in features.groupby('extractor'):
        tab = tab.xs(ext, level='extractor').xs('clinical', level='timepoint')
        
        rtab_path = util.working_path(f'lmm/{ext}.csv', write=True)
        #tab.to_csv(rtab_path)
        result_path = util.working_path(f'lmm/{ext}_cccs.csv', write=True)

        #util.r_run('lmm', (rtab_path, result_path))
        lmm_result = pandas.read_csv(result_path, index_col=0).reset_index(drop=True)
        lmm_result['extractor'] = ext
        lmm_result = lmm_result.set_index(['extractor', 'feature']).dropna(axis='index')
        lmm_results.append(lmm_result)

    cccs = pandas.concat(lmm_results, axis=0)
    cccs['roi'] = cccs.index.map(lambda s: 'liver' if s[1].startswith('liver') else 'tumor')
    cccs['feature_name'] = cccs.index.map(lambda s: s[1].replace('liver_', '').replace('tumor_', ''))
    cccs['feature_family'] = cccs['feature_name'].map(lambda s: s.split('_')[1])

    cccs.to_csv(util.working_path('lmm_cccs.csv', write=True))


def compute_harrels_per_features(features, outcome):
    from lifelines.utils import concordance_index
    outcome = outcome.loc[features.index]
    time = outcome['time']
    event = outcome['event']

    s = {}
    for name, vals in features.items():
        s[name] = concordance_index(time[vals.index], vals, event[vals.index])

    result = pandas.Series(s, name='c-index')
    result.index.name = 'feature'
    return result
        
    
@entry.point
def univariate_survival(args):
    util = Utils.from_file(args.conf)

    surv_features = util.load_survival_tables()
    surv_outcomes = util.univariate_survival_outcome()

    tabs = []
    for ext, features in surv_features.groupby('extractor'):
        features = features.xs(ext, level='extractor')
        cis = compute_harrels_per_features(features, surv_outcomes)
        cis = cis.to_frame('c-index')
        cis['extractor'] = ext
        cis = cis.set_index('extractor', append=True).reorder_levels(['extractor', 'feature'])

        tabs.append(cis)

    table = pandas.concat(tabs, axis=0)
    table['roi'] = table.index.map(lambda s: 'liver' if s[1].startswith('liver') else 'tumor')
    table['feature_name'] = table.index.map(lambda s: s[1].replace('liver_', '').replace('tumor_', ''))
    table['feature_family'] = table['feature_name'].map(lambda s: s.split('_')[1])
    table['somers'] = 2 * table['c-index'] - 1
    table['abs_somers'] = table['somers'].abs()
    table['abs_c-index'] = (table['abs_somers'] + 1) / 2

    table.to_csv(util.working_path('survival/univariate_c-indices.csv', write=True))

@entry.point
def multivariate_survival(args):
    from joblib import Parallel, delayed
    conf = Utils.from_file(args.conf)
    reps = conf.conf['multivariate_survival']['cv_repeats']

    results = Parallel(n_jobs=args.jobs, verbose=10)(delayed(_multivariate_survival_one_repeat)(conf, args, rep) for rep in range(reps))

    names = set()
    result_tables = []
    fsel_tables = []
    for metadata, result, fsel in results:
        metadata_name = f"{metadata['feature_set']}_{metadata['feature_selection']}_{metadata['feature_count']}"
        names.add(metadata_name)
        result_tables.append(result)
        fsel_tables.append(fsel)

    assert len(names) == 1
    metadata_name = names.pop()
    table = pandas.concat(result_tables, axis=0)
    fsel_table = pandas.concat(fsel_tables, axis=0)

    table.to_csv(conf.working_path(f"multivariate_survival/scores/{metadata_name}.csv", write=True), index=False)
    fsel_table.to_csv(conf.working_path(f'multivariate_survival/features/{metadata_name}.csv', write=True), index=False)

    

def _multivariate_survival_one_repeat(conf, args, repeat):
    def do_score(est, ind, X, y):
        X = X.iloc[ind]
        y = y.iloc[ind]
        sc = est.score(X, y)
        return sc
    def do_pred(est, ind, X):
        X = X.iloc[ind]
        sc = est.predict(X)
        return sc


    metadata = dict(
        feature_set=conf.conf['multivariate_survival']['feature_sets'][args.feature_set],
        feature_count=conf.conf['multivariate_survival']['feature_counts'][args.feature_count],
        feature_selection=conf.conf['multivariate_survival']['feature_selection'][args.feature_selection],
        cv_repeat = repeat
    )

    cv = conf.multivariate_cross_validation(args, repeat)
    model = conf.multivariate_survival_model(args)
    features = conf.multivariate_survival_features(args)
    outcomes = conf.multivariate_survival_outcome()

    try:
        res = cross_validate(model, features, outcomes, cv=cv, return_train_score=True, return_estimator=True, return_indices=True)
    except ValueError:
        print("Every fold failed! :(")
        sys.exit(0)
    train_scores = res['train_score']
    test_scores = res['test_score']

    def make_fsel_row(feature_set, fold):
        row = pandas.Series(0, index=features.columns)
        row[feature_set] = 1

        md = metadata.copy()
        md['cv_fold'] = fold
        md.update(row.to_dict())
        return md


    rows = []
    fsel_rows = []
    preds = []
    for fold, (est, trind, teind, trscore, tescore) in enumerate(zip(res['estimator'], res['indices']['train'], res['indices']['test'], train_scores, test_scores)):
        try:
            tepred = do_pred(est, teind, features)
        except AttributeError:
            tepred = []
        preds.append(tepred)
        feature_set = est.feature_selector_.get_selected(features).columns

        row = metadata.copy()
        row.update(dict(
            cv_fold=fold, score_type="inner", train=trscore, test=tescore
        ))
        rows.append(row)
        fsel_rows.append(make_fsel_row(feature_set, fold))

    preds = pandas.concat((preds), axis=0)
    outcomes_for_pred = outcomes.loc[preds.index]
    # Note: This handles the case with folds that failed to run. But we don't want to have too many of these, so it is import to check.
    outer_score = model.score_predictions(preds, outcomes_for_pred)
    row = metadata.copy()
    row.update(dict(
        cv_fold=-1, score_type="outer", train=-1, test=outer_score
    ))
    rows.append(row)

    table = pandas.DataFrame.from_records(rows)
    #table.to_csv(conf.working_path(f"multivariate_survival/scores/{metadata_name}.csv", write=True), index=False)

    fsel_table = pandas.DataFrame.from_records(fsel_rows)
    #fsel_table.to_csv(conf.working_path(f'multivariate_survival/features/{metadata_name}.csv', write=True), index=False)
    return metadata, table, fsel_table


@multivariate_survival.parser
def multivariate_survival_parser(parser):
    parser.add_argument('--feature_set', type=int, required=True)
    parser.add_argument('--feature_count', type=int, required=True)
    parser.add_argument('--feature_selection', type=int, required=True)
    parser.add_argument("--jobs", type=int, required=False, default=-1)


@entry.point
def multivariate_survival_args(args):
    util = Utils.from_file(args.conf)
    index = args.index
    conf = util.conf['multivariate_survival']
    lengths = [len(conf['feature_sets']), len(conf['feature_counts']), len(conf['feature_selection'])]

    total = numpy.product(lengths)
    assert index < total
    if index < 0:
        print(total)

    else:
        loop_sizes = numpy.cumprod(lengths[-1:0:-1])[::-1].tolist()+[1]
        arg_names = ["--feature_set", "--feature_count", "--feature_selection"]
        remainder = index
        results = []
        for name, length, loop_size in zip(arg_names, lengths, loop_sizes):
            value = remainder // loop_size
            remainder = remainder % loop_size

            assert value < length
            results.extend([name, str(value)])

        print(" ".join(results))

@multivariate_survival_args.parser
def _(parser):
    parser.add_argument("index", type=int)





        




        





    

@entry.point
def reproducibility(args):
    util = Utils.from_file(args.conf)




def main():
    entry.main()

if __name__=="__main__": 
    main()
