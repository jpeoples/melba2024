library(cccrm)

get_spacing_ccc_for_feature <- function (tab, feat){
	res <- cccvc(tab, feat, "patient", "spacing", covar="asir")
	return(res)
}

get_features_from_table <- function (tab) {
	cols <- colnames(tab)
	mask <- startsWith(cols, "liver") | startsWith(cols, "tumor")
	return(cols[mask])
}

get_all_ccc <- function (tab) {
	df <- data.frame(feature=character(), ccc=numeric(), Subject=numeric(), Observer=numeric(), Random=numeric())
	feats <- get_features_from_table(tab)
	for (i in 1:length(feats)) {
		f <- feats[[i]]
		#print(paste("   ", f))
		tryCatch({
			res <- get_spacing_ccc_for_feature(tab, f)
			new_row <- c(f, res$ccc["CCC"], res$vc)
			df[i, ] <- new_row
		}, error = function(e) { print("Error"); print(e) })
		
	}
	return(df)
}

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2) {
	stop("One argument -- the feature file descriptor -- must be included")
}

fdesc <- args[1]
input_file <- args[1]
output_file <- args[2]

print(paste("Reading features from", input_file))
features <- read.csv(input_file)

features$asir <- as.factor(features$asir)

r2d<-get_all_ccc(features)

print(paste("Writing CCCs to", output_file))
write.csv(r2d, output_file)

