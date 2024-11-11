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

# This is a handy object for creating scripts with multiple entry points that we call by name like
# python melba.py <entrypoint_name>
#
# These are created with this object by a decorator method
#
# @entry.point
# def entrypoint_name(args):
#     # code

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
        """Run an R script"""
        rpath = self.conf['r_path']
        scriptpath = self.conf['r_scripts'][script]
        arg_string = " ".join(args)

        os.system(f'"{rpath}" {scriptpath} {arg_string}')

    def data_path(self, path, prefix=None, external=False, write=False):
        """Helper function for paths in data folder"""
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
        """Ensure the parent directory of the specified path already exists"""
        dir = os.path.dirname(fpath)
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        return fpath

    def input_path(self, path):
        """Get a file from the input data directory"""
        return self.data_path(path, prefix='input')

    def working_path(self, path, write=False):
        """Get a file from the working data directory"""
        return self.data_path(path, prefix="working", write=write)

    def output_path(self, path, write=False):
        """Get a file from the output data directory"""
        return self.data_path(path, prefix="output", write=write)

    def load_repro_tables(self):
        """Load the repro tables for all extractors"""
        tables = {k: load_repro_table(self.data_path(p)) for k, p in self.conf['repro_tables'].items()}
        cat = []
        for ext, tab in tables.items():
            tab['extractor'] = ext
            tab = tab.set_index('extractor', append=True)
            tab = tab.reorder_levels(('extractor', 'patient', 'timepoint', 'spacing', 'asir'))
            cat.append(tab)

        return pandas.concat(cat, axis=0)

    def load_survival_table(self, feature_set):
        """Load survival table for a requested extractor"""
        assert feature_set in self.conf['survival_tables']
        p = self.data_path(self.conf['survival_tables'][feature_set])
        return load_survival_table(p)


    def load_survival_tables(self):
        """Load survival tables for all extractors"""
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
        """Load the clinical data for survival analysis"""
        table = pandas.read_csv(self.data_path(self.conf['survival_data']), index_col=0)
        return table

    def univariate_survival_outcome(self):
        """Get the survival event and time columns only for univariate analysis"""
        tab = self.load_survival_clin_table()
        conf = self.conf['univariate_survival']
        time = conf['time']
        event = conf['event']

        return pandas.DataFrame({"time": tab[time], "event": tab[event]})

    def multivariate_survival_outcome(self):
        """Get the survival event and time columns only for multiivariate analysis"""
        tab = self.load_survival_clin_table()
        conf = self.conf['multivariate_survival']
        time = conf['time']
        event = conf['event']

        return pandas.DataFrame({"time": tab[time], "event": tab[event]})

    def get_seed(self, repeat):
        """Deterministic function to get the nth seed"""
        seed = self.conf['multivariate_survival']['seed_generator_seed']
        rng = numpy.random.RandomState(seed)
        return rng.randint(2**31, size=(repeat+1,))[-1]

    def get_random_state(self, repeat):
        """Get a random state object from nth seed"""
        seed = self.get_seed(repeat)
        return numpy.random.RandomState(seed)

    def multivariate_survival_features(self, args):
        """Load features for multivariable survival modeling"""
        fs = self.conf['multivariate_survival']['feature_sets'][args.feature_set]
        if fs == 'all':
            features = self.load_survival_tables()
            features = blast_out_extractors(features)
        else:
            features = self.load_survival_table(fs)
        
        return features

    def multivariate_feature_selector(self, args):
        """Prepare feature selector for multivariable survival.
        
        This works with the supplied command line args for the model.

        As needed, it prepares the model to threshold based on CCC, apply the
        univariate thresholding based on C-index and p-value, and then run mRMR.
        """
        def cccs_for_feature_set(args):
            fs = self.conf['multivariate_survival']['feature_sets'][args.feature_set]
            cccs = pandas.read_csv(self.working_path('lmm_cccs.csv'), index_col=[0,1])

            if fs == "all":
                cccs = cccs['ccc']
                cccs.index = cccs.index.map(lambda s: f"{s[0]}_{s[1]}")
            else:
                cccs = cccs.loc[fs, 'ccc']

            return cccs

        
        fs_conf = self.conf['multivariate_survival']['feature_selection'][args.feature_selection]
        if "ccc_threshold" in fs_conf:
            cccs = cccs_for_feature_set(args)
            thresh = fs_conf['ccc_threshold']
            thresh_keep = cccs >= thresh
            thresh_keep = cccs.index[thresh_keep]
            selectors = [ListFeatSel(thresh_keep)]
        else:
            selectors = []
        
        univar = UnivarSigFeatSel(rel_thresh=self.conf['multivariate_survival']['ci_threshold'], p_thresh=self.conf['multivariate_survival']['p_threshold'])
        selectors.append(univar)

        selectors = tuple(selectors)
        return lambda n: FeatureSelectorChain(list(selectors)+[mRMRFeatSel(n, None)])

    def multivariate_survival_model(self, args):
        """Prepare the modeling object for multivariable survival analysis"""
        nfeat = self.conf['multivariate_survival']['feature_counts'][args.feature_count]
        fsel_constructor = self.multivariate_feature_selector(args)
        return MultivariateSurvival(nfeat, fsel_constructor)

    def multivariate_cross_validation(self, args, repeat):
        """Prepare the cross-validation object for multivariable analysis"""
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
    """Convert the long format with extractor in index, to a wide format with separate columns for each extractor/feature combination"""
    out = []
    for ext, tab in tab.groupby('extractor'):
        tab = tab.loc[ext]
        tab.columns = [f"{ext}_{c}" for c in tab.columns]
        out.append(tab)

    return pandas.concat(out, axis=1)

class CrossValidation:
    """Cross validation wrapper that works with y as a survival table (with columns 'time' and 'event')"""
    def __init__(self, cv):
        self.cv = cv

    def split(self, X, y, groups=None):
        return self.cv.split(X, y['event'], groups)
    

    def get_n_splits(self, X, y, groups=None):
        return self.cv.get_n_splits(X, y, groups)

class Normalizer:
    """Threshold low-variance features and convert to z-scores."""
    def compute_normalization(self, X):
        self.threshold_ = VarianceThreshold()
        X = self.threshold_.fit_transform(X)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X)

    def normalize(self, X):
        Xout = pandas.DataFrame(self.scaler_.transform(self.threshold_.transform(X)), index=X.index, columns=self.threshold_.get_feature_names_out(X.columns))
        return Xout

def surv_rel(X, y, feature_prefs=None):
    """A relevance score for survival data, always between 0.5 and 1.
    
    Based on Harrell's C-index. Negatively related features (C-index < 0.5)
    are converted to positive by converting the C-index to Somer's D (ranging from -1 to 1),
    then taking the absolute value, and converting back to Harrell's C.
    """
    ci = compute_harrels_per_features(X, y)
    dxy = 2 * ci - 1
    adxy = dxy.abs()
    oci = (adxy + 1) / 2

    if feature_prefs is not None:
        oci = (oci - oci.min()) / (oci.max() - oci.min())
        fp = (feature_prefs - feature_prefs.min()) / (feature_prefs.max() - feature_prefs.min())
        oci = oci * fp

    return oci

class UnivarSigFeatSel:
    def __init__(self, rel_thresh=0.55, p_thresh=0.1, relevance=surv_rel):
        self.rel_thresh=rel_thresh
        self.p_thresh = p_thresh
        self.relevance = relevance

    def select_features(self, X, y):
        rels = self.relevance(X, y)
        X = X.loc[:, rels > self.rel_thresh]
        to_keep = []
        for ft in X.columns:
            res = univar_cox(y, X[ft])
            if res['p-value'] < self.p_thresh:
                to_keep.append(ft)

        #X = X.loc[:, to_keep]
        print("Remaining after univar:", len(to_keep))
        self.selected_ = to_keep

    def get_selected(self, X):
        return X.loc[:, self.selected_]


class mRMRFeatSel:
    def __init__(self, nfeat=10, feat_prefs=None):
        self.nfeat=nfeat
        self.feat_prefs = feat_prefs

    def select_features(self, X, y):
        selected = mrmr_surv(X, y, self.nfeat, relevance=lambda X, y: surv_rel(X, y, feature_prefs=self.feat_prefs), show_progress=False, n_jobs=1)
        self.selected_ = selected

    def get_selected(self, X):
        return X.loc[:, self.selected_]

class FeatureSelectorChain:
    def __init__(self, selectors):
        self.selectors = selectors

    def select_features(self, X, y):
        for sel in self.selectors:
            sel.select_features(X, y)
            X = sel.get_selected(X)

        self.selected_ = X.columns

    def get_selected(self, X):
        return X.loc[:, self.selected_]

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

class ListFeatSel:
    def __init__(self, feature_list):
        self.feature_list = feature_list

    def select_features(self, X, y):
        self.selected_ = self.feature_list

    def get_selected(self, X):
        return X.loc[:, self.selected_]

def mrmr_surv(
        X, y, K,
        relevance = None, redundancy='c', denominator='mean',
        cat_features=None, cat_encoding='leave_one_out',
        only_same_domain=False, return_scores=False,
        n_jobs=-1, show_progress=True
):
    """Just a wrapper for mrmr_classif, but we pass the right relevance score (mrmr_surv_rel)"""
    return mrmr.mrmr_classif(X, y, K, relevance=relevance, redundancy=redundancy, denominator=denominator, cat_features=cat_features, cat_encoding=cat_encoding, only_same_domain=only_same_domain, return_scores=return_scores, n_jobs=n_jobs, show_progress=show_progress)


class MultivariateSurvival(BaseEstimator):
    """Multi-variable survival model implementation."""
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

def value_filter(features):
    """Remove a column if more than 50% of rows are equal.
    
    This is implemented to automatically detect and drop features like
    firstorder_Minimum, which because we are applying resegmentation to
    [-50,350], is almost always -50 (at least in the liver parenchyma). This
    doesn't play well with mRMR selection, which appears to frequently 
    select such features, even when they are not informative.
    """
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
    """Lee's concordance correlation coefficient"""
    np = numpy
    cov = np.cov(y_true, y_pred, ddof=0)
    mean_true=np.mean(y_true)
    mean_pred=np.mean(y_pred)

    var_true = cov[0,0]
    var_pred = cov[1,1]

    covar = cov[0,1]
    return 2 * covar / (var_true + var_pred + (mean_true-mean_pred)**2)

def parse_repro_caseid(caseid):
    """Extract patient, timepoint, slice thickness and asir from case id"""
    spl = caseid.split("_")
    asir = int(spl[-1])
    slice_thickness = int(spl[-2])/100
    timepoint = spl[-3]
    subject = "_".join(spl[:-3])

    return subject, timepoint, slice_thickness, asir

def expand_repro_index(features):
    """Parse case ids in index to a structured multiindex"""
    findex = features.index
    findex = [parse_repro_caseid(id) for id in findex]
    findex = pandas.MultiIndex.from_tuples(findex)
    features.index = findex
    features.index.names = ["patient", "timepoint", "spacing", "asir"]
    return features

def drop_diagnostics(features):
    """Drop diagnostic columns from pyRadiomics csv files"""
    to_drop = fnmatch.filter(features.columns, "*diagnostic*")
    return features.drop(columns=to_drop)

def load_repro_table(fname, keep_diag=False):
    """Load feature file from reproducibility data set"""
    tab = pandas.read_csv(fname, index_col=0)
    if not keep_diag:
        tab = drop_diagnostics(tab)
    tab = expand_repro_index(tab)


    return tab

def load_survival_table(fname, keep_diag=False):
    """Load feature file for survival data set (TCIA)."""
    tab = pandas.read_csv(fname, index_col=0)
    if not keep_diag:
        tab = drop_diagnostics(tab)
    return tab


@entry.add_common_parser
def common_parser(parser):
    parser.add_argument("--conf", required=False, default="conf.json")


def compute_pairwise_ccs(a, b):
    """Compute Lee's CCC for every column across two tables"""
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

    def make_fig(self, w=1, h=1, nrow=1, ncol=1, layout='constrained', **kwargs):
        plt = pyplot
        fig, ax = plt.subplots(nrow, ncol, figsize=(self.wmultiple*w, self.hmultiple*h), dpi=self.dpi, layout=layout, **kwargs)
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
    xorder = HUE_ORDER
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
        tab.to_csv(rtab_path)
        result_path = util.working_path(f'lmm/{ext}_cccs.csv', write=True)

        util.r_run('lmm', (rtab_path, result_path))
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
def cluster_figs(args):
    figs = FigSet()
    w = figs.wmultiple
    h = figs.hmultiple
    w *= 2/4
    h *= 2
    
    util = Utils.from_file(args.conf)
    cccs = pandas.read_csv(util.working_path('lmm_cccs.csv'), index_col=[0,1])
    cccs['feature_group'] = cccs['feature_family'].map(lambda s: 'First Order' if s=='firstorder' else s.upper())
    cis = pandas.read_csv(util.working_path('survival/univariate_c-indices.csv'), index_col=[0,1])
    groups = cccs.groupby('feature_name')['feature_group'].value_counts().reset_index('feature_group')['feature_group']


    def do_plot(ccc_map, groups, w, h):
        ugroups = groups.unique()
        colors = sns.color_palette()
        cmap = pandas.Series(colors[:len(ugroups)], index=ugroups)
        fcolors = ccc_map.index.map(groups).map(cmap)
        fcolors.name = "Feature Class"
        fig = sns.clustermap(ccc_map, col_cluster=True, metric='euclidean', method='ward', row_colors=fcolors, dendrogram_ratio=(0.2, 0.1), yticklabels=False, xticklabels=1, figsize=(w,h))
        return fig

    ccc_map = pandas.pivot_table(cccs, values='ccc', index='feature_name', columns=['extractor', 'roi']).dropna(axis=0)
    ccc_map.columns = ccc_map.columns.map(lambda s: f"{s[1][0]}-{s[0]}")
    ccc_map.columns.name = None
    ccc_map.index.name = "Feature"
    f = do_plot(ccc_map, groups, w*2, h)
    f.savefig(util.output_path("figs/ccc_cluster_roi_labelled.pdf", write=True))
    figs.figs.append(f)
    ccc_map = pandas.pivot_table(cccs, values='ccc', index='feature_name', columns=['extractor', 'roi']).dropna(axis=0)
    ccc_map.columns = ccc_map.columns.map(lambda s: s[0])
    ccc_map.columns.name = None
    ccc_map.index.name = "Feature"
    f = do_plot(ccc_map, groups, w*2, h)
    f.savefig(util.output_path("figs/ccc_cluster_no_roi.pdf", write=True))
    figs.figs.append(f)

    #do_plot(cis, groups)
    # NOTE 0.922 is an empirically derived ratio to get the heights of the actual clusters to roughly match with the CCC plot above (due to longer column labels..)
    lcis = cis.loc[cis['roi']=='liver']
    ccc_map = pandas.pivot_table(lcis, values='abs_c-index', index='feature_name', columns=['extractor']).dropna(axis=0)
    ccc_map.columns.name = None
    ccc_map.index.name = "Feature"
    f = do_plot(ccc_map, groups, w, h)
    f.savefig(util.output_path("figs/ci_cluster_liver.pdf", write=True))
    figs.figs.append(f)

    tcis = cis.loc[cis['roi']=='tumor']
    ccc_map = pandas.pivot_table(tcis, values='abs_c-index', index='feature_name', columns=['extractor']).dropna(axis=0)
    ccc_map.columns.name = None
    ccc_map.index.name = "Feature"
    f = do_plot(ccc_map, groups, w, h)
    f.savefig(util.output_path("figs/ci_cluster_tumor.pdf", write=True))
    figs.figs.append(f)
    #figs.set_dpi(72)

    fig, ax = figs.make_fig(1,1)
    sns.boxplot(cccs.rename(columns={"feature_group": "Feature Class"}), x='ccc', y='roi', hue='Feature Class', ax=ax)
    fig.savefig(util.output_path("figs/cluster_legend_source.pdf", write=True))
    fig.set_dpi(72)

    pyplot.show()

def zero_counts(extractors, rois):
    ix = pandas.MultiIndex.from_tuples([(ex, roi) for ex in extractors for roi in rois], names=["extractor", "roi"])
    return pandas.Series(0, index=ix)

HUE_ORDER = ["L2i", "S2i", "A2", "L2", "S2", "A3", "L3", "S3"]
def plot_top_rank_counts(table, ax=None, col="ccc", hue_order=HUE_ORDER):
    table = table.copy()
    table['rank'] = table.groupby('feature')[col].rank(ascending=True, method='max')#.groupby(['extractor')['ccc'].mean().sort_values()
    zeros = zero_counts(hue_order, ["tumor", "liver"])
    tab = table.groupby(["extractor", "roi"])["rank"].value_counts().xs(len(hue_order), level='rank')
    zeros.update(tab)
    tab = zeros.to_frame("count").loc[hue_order]
    sns.barplot(tab, x="extractor", y="count", hue="extractor", hue_order=hue_order, errorbar=('pi', 100), ax=ax)
    sns.pointplot(tab, x="extractor", y="count", hue="roi", hue_order=['tumor', 'liver'], dodge=0.3, linestyle='none', color='k', markers=['4', '3'], ax=ax)

def pareto_front(tab, a, b):
    acol = tab[a]
    bcol = tab[b]

    front = []
    for ix, row in tab.iterrows():
        av = row[a]
        bv = row[b]

        amask = acol >= av
        bmask = bcol >= bv
        emask = (acol == av) & (bcol == bv)

        com_mask = amask & bmask
        if com_mask.sum() == 1:
            front.append(ix)
        else:
            if (com_mask).equals(emask):
                front.append(ix)

    return front

@entry.point
def pareto_figs(args):
    util = Utils.from_file(args.conf)

    cccs = pandas.read_csv(util.working_path('lmm_cccs.csv'), index_col=[0,1])
    cis = pandas.read_csv(util.working_path('survival/univariate_c-indices.csv'), index_col=[0,1])

    figs = FigSet()

    fig_a, axs = figs.make_fig(1, 2, 2, 2, sharex=True, sharey=False)

    all_data = cccs.join(cis, how='inner', rsuffix='_cis')

    plot_top_rank_counts(all_data, ax=axs[0,0], col="ccc")
    axs[0,0].set_title("Highest CCC")
    axs[0,0].legend()
    #fig.savefig(util.output_path("figs/pareto_ccc_winner.pdf", write=True))

    #fig, ax = figs.make_fig(0.5, 1)
    plot_top_rank_counts(all_data, ax=axs[0,1], col="abs_c-index")

    axs[0,1].set_title("Highest C-index")
    axs[0,1].legend()
    #fig.savefig(util.output_path("figs/pareto_ci_winner.pdf", write=True))


    pareto_inds = pareto_front(all_data, 'abs_c-index', 'ccc')
    all_data['pareto'] = 'non-pareto'
    all_data.loc[pareto_inds, 'pareto'] = 'pareto'
    pareto = all_data.loc[pareto_inds]

    fig, ax = figs.make_fig(1, 2)
    hue_order = HUE_ORDER
    style_order = ["tumor", "liver"]
    sns.scatterplot(all_data, x='abs_c-index', y = 'ccc', hue='extractor', hue_order=hue_order, style='roi', style_order=style_order, markers=["4", "3"], alpha=0.4, linewidths=2.5, legend=False, s=100) #style='pareto', style_order=['non-pareto', 'pareto'])
    sns.scatterplot(pareto.reset_index().rename(columns={"extractor": "Extractor", "roi": "ROI"}), x='abs_c-index', y = 'ccc', hue='Extractor', legend=True, hue_order=hue_order, style='ROI', style_order=style_order, markers=["4", "3"], linewidths=2.5, s=100) #style='pareto', style_order=['non-pareto', 'pareto'])
    sns.scatterplot(pareto, x='abs_c-index', y='ccc', markers="o", facecolors="none", edgecolors="r", s=150, linewidths=2)
    ax.set_xlabel("C-index")
    ax.set_ylabel("CCC")
    fig.savefig(util.output_path("figs/pareto_front.pdf", write=True))


    #fig, ax = figs.make_fig(0.5, 1)
    zeros = zero_counts(hue_order, ["tumor", "liver"])
    counts = pareto.value_counts(["extractor", 'roi'])
    zeros.update(counts)
    counts = zeros
    counts = counts.to_frame("count").loc[hue_order]
    sns.barplot(counts, x="extractor", y="count", hue="extractor", hue_order=hue_order, ax=axs[1,1])
    sns.pointplot(counts, x="extractor", y="count", hue="roi", hue_order=['tumor', 'liver'], dodge=0.3, linestyle='none', color='k', markers=['4', '3'], ax=axs[1,1])
    axs[1,1].legend()
    axs[1,1].set_title("Pareto Front")
    axs[1,1].set_yticks([0, 1, 2, 3, 4])
    #fig.savefig(util.output_path("figs/pareto_overall_count.pdf", write=True))


    #fig, ax = figs.make_fig(0.5, 1)
    import collections
    counts = zero_counts(hue_order, ['tumor', 'liver'])
    for ft, data in all_data.groupby("feature"):
        pf = pareto_front(data, 'ccc', 'abs_c-index')
        for ex, f in pf:
            roi = f.split("_")[0]
            counts[(ex, roi)] += 1
    countt = pandas.Series(counts, name="count")
    countt.index.names = ["extractor", 'roi']
    countt = countt.to_frame('count').loc[hue_order]
    sns.barplot(countt, x='extractor', y='count', hue="extractor", hue_order=hue_order, ax=axs[1,0])
    sns.pointplot(countt, x="extractor", y="count", hue="roi", hue_order=['tumor', 'liver'], dodge=0.3, linestyle='none', color='k', markers=['4', '3'], ax=axs[1,0])
    axs[1,0].set_title("Per-feature Pareto Front")
    axs[1,0].legend()

    [ax.set_ylabel("Feature Count") for ax in axs.flatten()]
    [ax.set_xlabel("Extractor") for ax in axs.flatten()]
    [ax.grid(axis='y') for ax in axs.flatten()]
    fig_a.savefig(util.output_path("figs/pareto_extractor_breakdown.pdf", write=True))

    figs.set_dpi(72)
    pyplot.show()


    print_cols = ["roi", "feature_family", "feature_name", "ccc", "abs_c-index"]
    renamer = {"ccc": "CCC", "abs_c-index": "C-index", "feature_family": "Class", "feature_name": "Name", "roi": "ROI", "extractor": "Extractor"}
    print_ready = pareto.loc[[h for h in hue_order if h in pareto.index.get_level_values("extractor")], print_cols].reset_index().drop(columns=["feature"]).rename(columns=renamer)
    print_ready["Name"] = print_ready["Name"].map(lambda s: s.split("_")[-1])
    class_mapper = dict(firstorder="First order", glcm="GLCM", gldm="GLDM", glszm="GLSZM", glrlm="GLRLM", ngtdm="NGTDM")
    print_ready["Class"] = print_ready["Class"].map(class_mapper)
    roimapper = dict(liver="Liver", tumor="Tumor")
    print_ready["ROI"] = print_ready["ROI"].map(roimapper)

    print(print_ready)
    print(print_ready.set_index(["Extractor", "ROI"], drop=True).to_latex(index=True, sparsify=True, longtable=False, float_format="{:0.4f}".format))
    with open(util.output_path("tables/pareto_features.txt", write=True), "w") as f:
        print(print_ready.set_index(["Extractor", "ROI"], drop=True).to_latex(index=True, sparsify=True, longtable=False, float_format="{:0.4f}".format), file=f)




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
        metadata_name = f"{metadata['feature_set']}_{int(metadata['ccc_threshold']*100):02d}_{metadata['feature_count']}"
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
        ccc_threshold=conf.conf['multivariate_survival']['feature_selection'][args.feature_selection].get("ccc_threshold", 0.0),
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
        raise

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
def multivariate_figs(args):
    import seaborn as sns
    util = Utils.from_file(args.conf)

    scores = pandas.read_csv(util.working_path('multivariate_survival/scores.csv'), index_col=[0,1,2,3,4])
    outer = scores.loc[scores['score_type']=='outer']
    inner_mns = scores.loc[scores['score_type']=='inner'].groupby(['feature_set', 'feature_count', 'ccc_threshold', 'cv_repeat']).mean(numeric_only=True)

    figs = FigSet()
    fig, ax = figs.make_fig(1,1)

    count = 4
    cmp_threshold=0.85

    
    # NOTE: This ordering comes from the clustering ordering (plus all at the front)
    set_order = ["all"] + HUE_ORDER

    tab = outer.xs(count, level='feature_count').copy()
    itab = inner_mns.xs(count, level='feature_count')
    tab = tab.loc[(tab.index.get_level_values('ccc_threshold') == 0) | (tab.index.get_level_values('ccc_threshold')==cmp_threshold)]
    train_vals = itab.loc[tab.index, 'train']
    tab['CCC'] = tab.index.get_level_values('ccc_threshold').map(lambda c: f"CCC={c:0.02f}")
    tab['CCC'] = tab.index.get_level_values('ccc_threshold').map(lambda c: f"CCC={c:0.02f}")
    tab['train'] = train_vals

    tab = tab.rename(columns=dict(feature_set="Extractor", test="Test C-index", train="Train C-index"))
    sns.boxplot(tab, x='feature_set', y='Test C-index', hue='CCC', ax=ax, legend=False, order=set_order)
    sns.pointplot(tab, x='feature_set', y='Train C-index', hue='CCC', errorbar='pi', linestyle='--',  ax=ax, legend=False)
    ax.set_ylim(0.5, 0.68)
    ax.set_ylabel("C-index")
    ax.set_xlabel("Extractor")
    ax.grid(visible=True, axis='y')
    fig.savefig(util.output_path("figs/boxplot_4_85.pdf"))

    # These are just to get the legends out of!
    fig, ax = figs.make_fig(1,1)
    tabx = tab.rename(columns=dict(CCC="Test"))
    sns.boxplot(tabx, x='feature_set', y='Test C-index', hue='Test',  legend=True, ax=ax)
    fig.savefig(util.output_path("figs/boxplot_4_85_test_legend.pdf"))
    fig, ax = figs.make_fig(1,1)
    tabx = tab.rename(columns=dict(CCC="Train"))
    sns.pointplot(tabx, x='feature_set', y='Train C-index', hue='Train', linestyle='--', legend=True, ax=ax)
    fig.savefig(util.output_path("figs/boxplot_4_85_train_legend.pdf"))


    # Generate table of top 10 performers
    t = outer.droplevel('cv_fold')
    means = t.groupby(['feature_set', 'feature_count', 'ccc_threshold'])['test'].mean()#.sort_values(ascending=False)[:10]
    cis = t.groupby(['feature_set', 'feature_count', 'ccc_threshold'])['test'].quantile([0.025, 0.975]).to_frame()
    cis.index.names = ['feature_set', 'feature_count', 'ccc_threshold', 'quantile']
    cis = pandas.pivot_table(cis, values='test', index=['feature_set', 'feature_count', 'ccc_threshold'], columns='quantile')

    top_means = means.sort_values(ascending=False)[:10].to_frame('mean')
    top_index = top_means.index
    tops = top_means.join(cis, how='left')


    # Load the feature selection counts
    path = util.working_path("multivariate_survival/roi_feature_counts.csv")
    if os.path.exists(path):
        selected_features = pandas.read_csv(path, index_col=[0,1,2,3,4])
    else:
    # NExt we load the feature selection data and compute some summary statistics
        features_all = pandas.read_csv(util.working_path("multivariate_survival/features_all.csv"), index_col=[0,1,2,3,4])
        features_single = pandas.read_csv(util.working_path("multivariate_survival/features_single.csv"), index_col=[0,1,2,3,4])

        unique_features = sorted(set(features_all.columns) | set(features_single.columns))
        rows = []
        for feat in unique_features:
            roi = 'liver' if 'liver' in feat else 'tumor'
            row = dict(feature=feat, liver=roi=='liver', tumor=roi=='tumor')
            rows.append(row)
        feature_data_table = pandas.DataFrame(rows).set_index('feature')
        feature_data_table_all = feature_data_table.loc[features_all.columns]
        feature_data_table_single = feature_data_table.loc[features_single.columns]

        def selected_features(row, table):
            tab = table.loc[row==1]
            return tab.sum()
            

        def selected_features_all(row):
            return selected_features(row, feature_data_table_all)

        def selected_features_single(row):
            return selected_features(row, feature_data_table_single)
        print("Tabulating selected featuers all")
        selected_features_all = features_all.apply(selected_features_all, axis=1)
        print("Tabulating selected featuers single")
        selected_features_single = features_single.apply(selected_features_single, axis=1)
        selected_features = pandas.concat((selected_features_all, selected_features_single), axis=0)
        selected_features.to_csv(path, index=True)

    total_selected = selected_features['liver'] + selected_features['tumor']
    norm_selected = selected_features.apply(lambda x: x / total_selected)
    mean_counts = norm_selected.groupby(["feature_set", "feature_count", "ccc_threshold"]).mean()

    long_counts = mean_counts.reset_index(drop=False).melt(id_vars=["feature_set", "feature_count", "ccc_threshold"], value_vars=['liver', 'tumor'], var_name='roi', value_name='count')
    long_counts = long_counts.loc[long_counts['roi']=='liver']
    #pivot = pandas.pivot_table(long_counts, index=['feature_set'], columns=['ccc_threshold', 'feature_count'], values='count')
    #print(pivot)


    top_indices = list(tops.index.to_flat_index())
    perc_features_drawn_from_liver = long_counts.set_index(["feature_set", "feature_count", "ccc_threshold"]).loc[top_indices, "count"]
    print(perc_features_drawn_from_liver)


    formatted = tops.apply(lambda row: f"{row['mean']:0.03f} ({row[0.025]:0.03f}--{row[0.975]:0.03f})", axis=1)
    formatted.index.names = ["Extractor", "Feature Count", "$\mathrm{CCC}_t$"]
    formatted = formatted.to_frame("Harrel's C-index")
    perc_features_drawn_from_liver.index.names = formatted.index.names
    formatted["Prop. Liver Features"] = perc_features_drawn_from_liver
    with open(util.output_path("tables/top_multivar_performers.txt", write=True), "w") as f:
        print(formatted.reset_index().to_latex(index=False,float_format="%0.02f"), file=f)





    # The full figure.
    # NOTE: For the feature count CIs, I have computed this based on getting the 0.025 and 0.975 quantiles of the count of selected features
    # This is basically just to give a sense of where teh number of features stops increasing for each plot. NOTE:
    # Originally I computed them for every feature set, ccc_threshold, and feature count. Then I realized that feature_count=64 gives all the needed info
    # (For any of these percentiles, if it is < requested feature count, it is the same for all, so feature_count=64 gives the max possible)
    featsel = pandas.read_csv('data/working/multivariate_survival/features_single.csv', index_col=[0,1,2,3,4])
    selcounts = featsel.sum(axis=1)
    featsel = pandas.read_csv('data/working/multivariate_survival/features_all.csv', index_col=[0,1,2,3,4])
    selcounts = pandas.concat((selcounts, featsel.sum(axis=1)), axis=0)
    #threshed = selcounts < selcounts.index.get_level_values('feature_count')
    #thsel = selcounts[threshed]
    percs = selcounts.groupby(['feature_set', 'ccc_threshold', 'feature_count']).describe(percentiles=[0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975])#.to_csv("check_feature_counts.csv")
    ix = pandas.IndexSlice
    feature_count_cis = percs.loc[ix[:, :, 64], ["2.5%", "97.5%"]]
    feature_count_cis["97.5%"][feature_count_cis["97.5%"] == 64.0] = 100
    feature_count_cis["2.5%"][feature_count_cis["2.5%"] == 64.0] = 100
    feature_count_cis = feature_count_cis.xs(64, level="feature_count")

    fig, axsouter = figs.make_fig(2, 3, nrow=9, ncol=5, sharex=True, sharey=True)

    sets=set_order

    outer_loop='feature_set'
    xaxis='feature_count'
    feature_counts = [1, 2, 4, 8, 16, 32, 64]

    for ixx, st in enumerate(sets):
        tab = outer.xs(st, level=outer_loop)
        itab = inner_mns.xs(st, level=outer_loop).copy()
        axs = axsouter[ixx]
        #fig, axs = pyplot.subplots(1,5, figsize=(20,4), sharey=True)

        tab = tab.drop(columns=['score_type'])
        tab = tab.xs(-1, level='cv_fold')
        tab['type'] = 'test'
        itab['type'] = 'train'
        itab = itab.rename(columns=dict(train='c-index', test='none'))
        tab = tab.rename(columns=dict(train='none', test='c-index'))
        newtab = pandas.concat((tab, itab), axis='rows')

        for (name, tab), ax in zip(newtab.groupby('ccc_threshold'), axs):
            sns.pointplot(tab, x=xaxis, y='c-index', hue='type', linestyles=['solid', 'dashed'], errorbar='pi', estimator='mean', ax=ax, native_scale=True, legend=False)
            ax.semilogx(base=2)
            ci = feature_count_cis.loc[(st, name)]
            lower, upper = ci['2.5%'], ci['97.5%']
            xr, yr = ax.get_xlim(), ax.get_ylim()

            y = [0, 1]
            x1 = [lower, lower]
            x2 = [upper, upper]
            color=[0.1, 0.1, 0.1, 0.1]
            ax.fill_betweenx(y, x1, x2, color=color)
            ax.set_xlim(xr)
            ax.set_ylim(yr)
            ax.set_ylim((0.48, 0.77))

            #ax.set_xscale('log', base=2)
            #ax.set_title(f"{st}, {name}")
            ax.set_xticks(feature_counts)
            #ax.legend(handlelength=3)
            from matplotlib.ticker import StrMethodFormatter
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:0.0f}"))
            if name == 0.0:
                ax.set_ylabel(f"{st}\nC-index")
            if ixx == 0:
                ax.set_title(f"$\mathrm{{CCC}} \geq {name:0.02f}$")
            if ixx == 8:
                ax.set_xlabel("Feature Count")

        [ax.grid(visible=True) for ax in axs]

    fig.savefig(util.output_path("figs/multivariate_full_summary.pdf", write=True))

    figs.set_dpi(72)
    pyplot.show()
    
@entry.point
def generate_additional_tables(args):
    util = Utils.from_file(args.conf)

    features = util.load_repro_tables()

    def class_and_name(x):
        spl = x.split("_")
        name = spl[-1]
        klass = spl[-2]
        return klass, name
    all_features = pandas.DataFrame(sorted(set(features.columns.map(class_and_name))), columns=["Feature Class", "Name"])
    assert all_features.shape[0] == 93

    names = dict(
        firstorder = "First order",
        glcm="GLCM",
        ngtdm="NGTDM",
        gldm="GLDM",
        glrlm="GLRLM",
        glszm="GLSZM"
    )
    all_features["Feature Class"] = all_features["Feature Class"].map(names)
    
    # Generate the counts
    print_ready = all_features.groupby("Feature Class").count().loc[list(names.values())].reset_index().rename(columns={"Name": "Count"})
    grand_total = pandas.DataFrame({"Feature Class": ["Total"], "Count": [print_ready["Count"].sum()]})
    styler = print_ready.style.concat(grand_total.style.set_properties(**{"font-weight": "bold"}))
    print(styler.hide(axis='index').to_latex(convert_css=True, hrules=True))
    with open(util.output_path("tables/feature_counts_by_class.txt", write=True), "w") as f:
        print(styler.to_latex(convert_css=True, hrules=True), file=f)

    # Generate the name lists
    #max_count = print_ready["Count"].max()
    #all_features["number"] = all_features.groupby("Feature Class").rank()

    list_table = all_features.set_index(["Feature Class", "Name"])
    list_table.columns.name = None
    list_table.index.name = None
    print(list_table)
    with open(util.output_path("tables/list_all_features.txt", write=True), "w") as f:
        print(list_table.to_latex(index=True, sparsify=True, longtable=True, label="tab:feature_list", caption="All features from all feature classes. Detailed feature definitions can be found in the \\texttt{pyradiomics} documentation. \\url{https://pyradiomics.readthedocs.io/en/latest/features.html}"), file=f)


@entry.point
def final_statistics(args):
    util = Utils.from_file(args.conf)
    cccs = pandas.read_csv(util.working_path("lmm_cccs.csv"), index_col=0)

    ccc_liver = cccs.loc[cccs['roi']=='liver']
    ccc_tumor = cccs.loc[cccs['roi']=='tumor']

    ccc_liver = ccc_liver.reset_index(drop=False).set_index(["extractor", "feature_name"])
    ccc_tumor = ccc_tumor.reset_index(drop=False).set_index(["extractor", "feature_name"])

    shared_index = ccc_liver.index.intersection(ccc_tumor.index)
    print(ccc_liver.shape[0], ccc_tumor.shape[0], len(shared_index))
    ccc_liver = ccc_liver.loc[shared_index]
    ccc_tumor = ccc_tumor.loc[shared_index]
    
    results = {}
    for ext, ltab in ccc_liver.groupby("extractor"):
        ttab = ccc_tumor.loc[ext]
        diffs = ltab['ccc'] - ttab['ccc']
        med_dif = diffs.median()
        med_l = ltab['ccc'].median()
        med_t = ttab['ccc'].median()
        sr_result = wilcoxon(ltab['ccc'].to_numpy(), ttab['ccc'].to_numpy(), alternative='less')
        results[ext] = dict(p=sr_result.pvalue, med_diff=med_dif, med_liver=med_l, med_tumor = med_t)

    results = pandas.DataFrame.from_dict(results, orient='index')
    print(results)
    results = results['p']

    print(f"p-value range for liver ccc < tumor ccc: [{results.min()}, {results.max()}]")


    path = util.working_path("multivariate_survival/roi_feature_counts.csv")
    if os.path.exists(path):
        selected_features = pandas.read_csv(path, index_col=[0,1,2,3,4])
    else:
    # NExt we load the feature selection data and compute some summary statistics
        features_all = pandas.read_csv(util.working_path("multivariate_survival/features_all.csv"), index_col=[0,1,2,3,4])
        features_single = pandas.read_csv(util.working_path("multivariate_survival/features_single.csv"), index_col=[0,1,2,3,4])

        unique_features = sorted(set(features_all.columns) | set(features_single.columns))
        rows = []
        for feat in unique_features:
            roi = 'liver' if 'liver' in feat else 'tumor'
            row = dict(feature=feat, liver=roi=='liver', tumor=roi=='tumor')
            rows.append(row)
        feature_data_table = pandas.DataFrame(rows).set_index('feature')
        feature_data_table_all = feature_data_table.loc[features_all.columns]
        feature_data_table_single = feature_data_table.loc[features_single.columns]

        def selected_features(row, table):
            tab = table.loc[row==1]
            return tab.sum()
            

        def selected_features_all(row):
            return selected_features(row, feature_data_table_all)

        def selected_features_single(row):
            return selected_features(row, feature_data_table_single)
        print("Tabulating selected featuers all")
        selected_features_all = features_all.apply(selected_features_all, axis=1)
        print("Tabulating selected featuers single")
        selected_features_single = features_single.apply(selected_features_single, axis=1)
        selected_features = pandas.concat((selected_features_all, selected_features_single), axis=0)
        selected_features.to_csv(path, index=True)

    total_selected = selected_features['liver'] + selected_features['tumor']
    norm_selected = selected_features.apply(lambda x: x / total_selected)
    mean_counts = norm_selected.groupby(["feature_set", "feature_count", "ccc_threshold"]).mean()

    long_counts = mean_counts.reset_index(drop=False).melt(id_vars=["feature_set", "feature_count", "ccc_threshold"], value_vars=['liver', 'tumor'], var_name='roi', value_name='count')
    long_counts = long_counts.loc[long_counts['roi']=='liver']
    print(long_counts)
    #pivot = pandas.pivot_table(long_counts, index=['feature_set'], columns=['ccc_threshold', 'feature_count'], values='count')
    #print(pivot)

    #sns.heatmap(pivot, vmin=0, vmax=1)
    #pyplot.show()
    best_perf_models = [
        ("L2i", 4, 0),
        ("all", 4, 0.85)
    ]

    perc_features_drawn_from_liver = long_counts.set_index(["feature_set", "feature_count", "ccc_threshold"]).loc[best_perf_models, "count"]
    print(perc_features_drawn_from_liver)
    

@entry.point
def generate_feature_visualizations(args):
    util = Utils.from_file(args.conf)
    conf = util.conf['feature_visualization']

    TCIA_ROOT = conf.get("tcia_root")

    import os
    import json
    import SimpleITK as sitk



    with open(conf.get("repro_dataset_file"), "r") as f:
        repro_dataset = json.load(f)

    with open(conf.get("surv_dataset_file"), "r") as f:
        surv_dataset = json.load(f)

    def get_components(mask):
        f = sitk.ConnectedComponentImageFilter()
        labeled = f.Execute(mask)
        count = f.GetObjectCount()
        return count, labeled

    def get_largest_component(mask, l):
        mask = mask == l
        count, labeled = get_components(mask)
        lss = sitk.LabelShapeStatisticsImageFilter()
        lss.Execute(labeled)
        assert lss.GetNumberOfLabels() == count
        maxv = 0
        biggest = 0
        for ix in range(1, count+1):
            v = lss.GetNumberOfPixels(ix)
            if v > maxv:
                maxv = v
                biggest = ix

        info = dict(count=count, labeled_components=labeled, filter=lss, label=biggest)
        
        return labeled == biggest, info

    # These functions both return the bbox as a 2xd array, where first row is start,
    # and second row is end. "End" here is the _last_ value, not the last + 1 like is
    # usually used for python slicing.
    def bbox_from_LabelShapeStatisticsImageFilter(filter, label):
        # LabelShapeStatisticsImageFilter returns the bbox as
        # [xstart, ystart, ztart, xsize, ysize, zsize]
        bbx = numpy.array(filter.GetBoundingBox(label))
        dim = int(len(bbx) / 2)
        bbx = bbx.reshape((2,dim))
        bbx[1] = bbx[0] + bbx[1] - 1
        return bbx

    def bbox_from_LabelStatisticsImageFilter(filter, label):
        # LabelStatisticsImageFilter returns the bbox as
        # [xstart, xend, ystart, yend, zstart, zend]
        bounding_box = numpy.array(filter.GetBoundingBox(label))
        dim = int(len(bounding_box)/2)
        return bounding_box.reshape((dim, 2)).transpose()


    def bounding_box(mask=None, img=None, filter=None, label=None, im_mode=None):
        if mask is not None:
            assert filter is None
            if img is not None:
                filter = sitk.LabelStatisticsImageFilter()
                filter.Execute(img, mask)
                im_mode = True
            else:
                filter = sitk.LabelShapeStatisticsImageFilter()
                filter.Execute(mask)
                im_mode = False

        if label is None:
            label=1

        if im_mode is None:
            im_mode = isinstance(filter, sitk.LabelStatisticsImageFilter)
        
        if im_mode:
            bbox = bbox_from_LabelStatisticsImageFilter(filter, label)
        else:
            bbox = bbox_from_LabelShapeStatisticsImageFilter(filter, label)
        
        info = dict(filter=filter, mask=mask, img=img, label=label, im_mode=im_mode)
        return bbox, info

    def crop(img, pad=0, bb=None):
        if bb is None:
            bb, _ = bounding_box(img)

        size = numpy.array(img.GetSize())
        l=numpy.maximum(bb[0] - pad, 0)
        u = numpy.minimum(bb[1] + pad + 1, size)
        if bb.shape[1] == 3:
            cropped = img[l[0]:u[0], l[1]:u[1], l[2]:u[2]]
        elif bb.shape[1] == 2:
            cropped = img[l[0]:u[0], l[1]:u[1]]

        info = dict(bounding_box=bb)
        return cropped, info

    def load_repro_image(patient, st, asir, msk, padding=10):
        key = f"{patient}_clinical_{int(st*100)}_{asir:02d}"
        im = repro_dataset[key]['image']
        seg = repro_dataset[key]['segmentation']

        rim = sitk.ReadImage(im)
        rseg = sitk.ReadImage(seg)

        if msk=='liver':
            rseg = rseg == 1
            bb, _ = bounding_box(mask=rseg, img=rim)
        else:
            rseg, info = get_largest_component(rseg, 2)
            bb, _ = bounding_box(filter=info['filter'], label=info['label'])
            
        rim, _ = crop(rim, bb=bb, pad=padding)
        rseg, _ = crop(rseg, bb=bb, pad=padding)

        return rim, rseg

    def load_surv_image(patient, roi='liver', padding=10):
        im = surv_dataset[patient]['image']
        seg = surv_dataset[patient]['segmentation'][roi]

        rim = sitk.ReadImage(os.path.join(TCIA_ROOT, im))
        rseg = sitk.ReadImage(os.path.join(TCIA_ROOT, seg))
        rim, rseg = match_image_and_segmentation(rim, rseg, expand_mask=True)


        bb, _ = bounding_box(mask=rseg) # seg is not always same size as im!, img=rim)
        rim, _ = crop(rim, bb=bb, pad=padding)
        rseg, _ = crop(rseg, bb=bb, pad=padding)
        
        return rim, rseg

    def mod_by_array(img):
        assert isinstance(img, sitk.Image)
        arr = sitk.GetArrayFromImage(img)

        def arr2img(arr):
            imout = sitk.GetImageFromArray(arr)
            imout.CopyInformation(img)
            return imout
        
        return arr, arr2img

    def im_information_agrees(ima, imb):
        attribs = (
            lambda x: x.GetSpacing(),
            lambda x: x.GetDirection(),
            lambda x: x.GetOrigin(),
            lambda x: x.GetSize(),
        )

        for a in attribs:
            if not numpy.allclose(a(ima), a(imb)):
                return False

        return True

    def expand_subsized_segmentation(ct, m):
        assert numpy.allclose(m.GetDirection(), ct.GetDirection(), atol=1e-3)
        assert numpy.allclose(m.GetSpacing(), ct.GetSpacing(), atol=1e-3)
        size = ct.GetSize()

        if ct.GetSize() == m.GetSize(): 
            # The images already are in same coord system and same number of slices, so we are fine
            assert numpy.allclose(ct.GetOrigin(), m.GetOrigin())
            return m
        
        # Mask volume is cropped, within CT volume, so we need to create our own image with the same size
        arr = numpy.zeros(size[::-1], dtype=numpy.uint8)
        mor = m.GetOrigin()
        mask_origin_index = ct.TransformPhysicalPointToIndex(mor)
        mask_bound = numpy.add(mor, numpy.multiply(m.GetSize(), m.GetSpacing()))
        mask_bound_index = ct.TransformPhysicalPointToIndex(mask_bound)
        
        # Copy the loaded mask into the CT.
        arr[mask_origin_index[2]:mask_bound_index[2], mask_origin_index[1]:mask_bound_index[1], mask_origin_index[0]:mask_bound_index[0]] = sitk.GetArrayViewFromImage(m)

        mout = sitk.GetImageFromArray(arr)
        mout.CopyInformation(ct)
        return mout

    def match_image_and_segmentation(img, msk, expand_mask=False):
        if im_information_agrees(img, msk):
            return img, msk
        if not expand_mask:
            origin = msk.GetOrigin()
            outer = msk.TransformIndexToPhysicalPoint(msk.GetSize())

            originindex = img.TransformPhysicalPointToIndex(origin)
            outerindex = img.TransformPhysicalPointToIndex(outer)
            assert (numpy.array(originindex) > 0).all()
            assert (numpy.array(outerindex) < img.GetSize()).all()
            adjusted_bb = numpy.array([originindex, outerindex])
            return crop(img, bb=adjusted_bb, pad=0), msk
        else:
            msk = expand_subsized_segmentation(img, msk)
            return img, msk

    def apply_segmentation(img, msk, bg=-1000, label=1):
        if im_information_agrees(img, msk):
            mskarr = sitk.GetArrayFromImage(msk)
            imarr, arr2im = mod_by_array(img)

            imarr[mskarr!=label] = bg
            return arr2im(imarr)
        else:
            assert False


    def apply_window_level(img, window, level, clamp=False):
        lower, upper = level - window / 2, level + window / 2

        imarr, arr2im = mod_by_array(img)
        imarr = (imarr - lower) / (upper - lower)
        if clamp:
            imarr[imarr < 0] = 0
            imarr[imarr > 1] = 1

        return arr2im(imarr)

    LIVER_WINDOW = 400
    LIVER_LEVEL = 40
    def prep_vis(im, seg):
        return apply_segmentation(apply_window_level(im, LIVER_WINDOW, LIVER_LEVEL, clamp=True), seg)

    def center_slice(im):
        sz = im.GetSize()
        return int(sz[2] / 2)

    def repro_slices(choices, roi, key='repro_high'):
        rim5, rseg5 = load_repro_image(choices[key][0], 5, 20, roi)
        rim3, rseg3 = load_repro_image(choices[key][0], 3.75, 20, roi)
        rim2, rseg2 = load_repro_image(choices[key][0], 2.5, 20, roi)

        rim5 = prep_vis(rim5, rseg5)
        rim3 = prep_vis(rim3, rseg3)
        rim2 = prep_vis(rim2, rseg2)

        s2 = center_slice(rim2)
        pp = rim2.TransformIndexToPhysicalPoint((0,0,s2))
        s5 = rim5.TransformPhysicalPointToIndex(pp)[2]
        s3 = rim3.TransformPhysicalPointToIndex(pp)[2]
        print(s2, s5, s3)

        rim5 = rim5[:, :, s5]
        rim3 = rim3[:, :, s3]
        rim2 = rim2[:, :, s2]

        rim5 = crop_slice(rim5)
        rim3 = crop_slice(rim3)
        rim2 = crop_slice(rim2)

        rim5 = sitk.GetArrayFromImage(rim5)
        rim3 = sitk.GetArrayFromImage(rim3)
        rim2 = sitk.GetArrayFromImage(rim2)

        return rim5, rim3, rim2

    def crop_slice(im, bg=-1000, pad=10):
        msk = im > bg
        bb, _ = bounding_box(msk, img=im)
        print(bb)
        im, _ = crop(im, bb=bb, pad=pad)
        return im

    def surv_slices(ch, roi, key):
        im, seg = load_surv_image(ch[key][0], roi)
        im = prep_vis(im, seg)
        s = center_slice(im)
        im = im[:, :, s]
        print("Surv im shape", im.GetSize())
        im = crop_slice(im)
        im = sitk.GetArrayFromImage(im)
        return im


    def plot_choice(choices):
        ext, feat = choices['feature']
        roi = feat.split("_")[0]
        ccc = choices['ccc']
        c_index = choices['c_index']
        print(roi)
        print(choices)
        rim5, rim3, rim2 = repro_slices(choices, roi, 'repro_high')
        rim5l, rim3l, rim2l = repro_slices(choices, roi, 'repro_low')
        sim = surv_slices(choices, roi, 'surv_high')
        siml = surv_slices(choices, roi, 'surv_low')

        from matplotlib import pyplot

        fig, axarr = pyplot.subplots(2, 4, layout='tight')

        ax = axarr[0]
        fig.suptitle(f'{ext}, {roi}, {"_".join(feat.split("_")[2:])}\nC-index={c_index:0.03f}, CCC={ccc:0.03f}')

        v5, v3, v2 = choices['repro_high'][1:] 
        vs = choices['surv_high'][1]
        ax[0].set_title(f"{v5:0.03f}")
        ax[1].set_title(f"{v3:0.03f}")
        ax[2].set_title(f"{v2:0.03f}")
        ax[3].set_title(f"{vs:0.03f}")
        ax[0].imshow(rim5, vmin=0, vmax=1, cmap='gray', aspect='equal')
        ax[1].imshow(rim3, vmin=0, vmax=1, cmap='gray', aspect='equal')
        ax[2].imshow(rim2, vmin=0, vmax=1, cmap='gray', aspect='equal')
        ax[3].imshow(sim, vmin=0, vmax=1, cmap='gray', aspect='equal')
        ax[0].set_frame_on(False)
        ax[1].set_frame_on(False)
        ax[2].set_frame_on(False)
        ax[3].set_frame_on(False)
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[2].set_xticks([])
        ax[3].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_yticks([])
        ax[2].set_yticks([])
        ax[3].set_yticks([])

        ax = axarr[1]
        v5, v3, v2 = choices['repro_low'][1:] 
        vs = choices['surv_low'][1]
        ax[0].set_title(f"{v5:0.03f}")
        ax[1].set_title(f"{v3:0.03f}")
        ax[2].set_title(f"{v2:0.03f}")
        ax[3].set_title(f"{vs:0.03f}")
        ax[0].imshow(rim5l, vmin=0, vmax=1, cmap='gray', aspect='equal')
        ax[1].imshow(rim3l, vmin=0, vmax=1, cmap='gray', aspect='equal')
        ax[2].imshow(rim2l, vmin=0, vmax=1, cmap='gray', aspect='equal')
        ax[3].imshow(siml, vmin=0, vmax=1, cmap='gray', aspect='equal')
        ax[0].set_frame_on(False)
        ax[1].set_frame_on(False)
        ax[2].set_frame_on(False)
        ax[3].set_frame_on(False)
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[2].set_xticks([])
        ax[3].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_yticks([])
        ax[2].set_yticks([])
        ax[3].set_yticks([])


    with open(conf['selection_file']) as f:
        objs = json.load(f)

    hhchoice = objs.get("high_high") 
    hlchoice = objs.get("high_low") 
    lhchoice = objs.get("low_high") 
    llchoice = objs.get("low_low") 

    plot_choice(hhchoice)
    plot_choice(hlchoice)
    plot_choice(lhchoice)
    plot_choice(llchoice)

    pyplot.show()


import ast
@entry.point
def image_spacing_histograms(args):
    util = Utils.from_file(args.conf)
    
    repro_tab_file = util.data_path(util.conf['repro_tables']['L2i'])
    surv_tab_file = util.data_path(util.conf['survival_tables']['L2i'])

    repro_tab = load_repro_table(repro_tab_file, keep_diag=True)
    surv_tab = load_survival_table(surv_tab_file, keep_diag=True)

    spacing_col = "liver_diagnostics_Image-original_Spacing"
    def get_spac_dim(tab, col, dim):
        return tab[col].map(lambda s: ast.literal_eval(s)[dim])

    repro_px = get_spac_dim(repro_tab, spacing_col, 0)
    repro_px = repro_px.xs((5, 20), level=("spacing", "asir"))
    surv_slice = get_spac_dim(surv_tab, spacing_col, 2)

    def make_plot(vals, ax):
        sns.histplot(vals, ax=ax)

    fig_set = FigSet()
    fig, ax = fig_set.make_fig(1, 1)
    make_plot(repro_px, ax)
    ax.set_xlabel("Pixel Spacing (mm)")
    ax.set_ylabel("Count")
    ax.grid(axis='y')
    fig.savefig(util.output_path("figs/repro_pixel_spacing_hist.pdf"))

    fig, ax = fig_set.make_fig(1, 1)
    make_plot(surv_slice, ax)
    ax.set_xlabel("Slice Thickness (mm)")
    ax.set_ylabel("Count")
    ax.grid(axis='y')
    fig.savefig(util.output_path("figs/surv_slice_thickness_hist.pdf"))
    
    fig_set.set_dpi(72)
    pyplot.show()







def main():
    entry.main()

if __name__=="__main__": 
    main()
