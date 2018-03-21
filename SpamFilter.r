#load library
library("tm")
library("SnowballC")
library("wordcloud")

#set the directory to home
setwd("/Users/Marisha/Downloads/dataset")

#train is the folder which is present in home directory and spam should be inside train
file_path_trainspam = file.path(".","train","spam")

#we set reader to readPlain as we dont know the format of the file
train_spam_Corpus <-Corpus(DirSource(file_path_trainspam), readerControl = list(reader = readPlain ))

file_path_trainham = file.path(".","train","ham")
train_ham_Corpus <-Corpus(DirSource(file_path_trainham), readerControl = list(reader = readPlain ))

file_path_testspam = file.path(".","test","spam")
test_spam_Corpus <-Corpus(DirSource(file_path_testspam), readerControl = list(reader = readPlain ))

file_path_testham = file.path(".","test","ham")
test_ham_Corpus <-Corpus(DirSource(file_path_testham), readerControl = list(reader = readPlain ))

#########
train_spam_Corpus <-VCorpus(VectorSource(train_spam_Corpus))
train_ham_Corpus <-VCorpus(VectorSource(train_ham_Corpus))
test_spam_Corpus <-VCorpus(VectorSource(test_spam_Corpus))
test_ham_Corpus <-VCorpus(VectorSource(test_ham_Corpus))
#########

#merge the train datasets
train_data <- c( train_spam_Corpus, train_ham_Corpus)

#merge the test datasets
test_data <- c(test_spam_Corpus, test_ham_Corpus)

train_data <-VCorpus(VectorSource(train_data))
test_data <-VCorpus(VectorSource(test_data))


#cleaning the data
corpus_train_clean <- tm_map(train_data, content_transformer(tolower))
corpus_test_clean <- tm_map(test_data, content_transformer(tolower))

#cleaning train spam ham only
corpus_train_spam_clean <- tm_map(train_spam_Corpus, content_transformer(tolower))
corpus_train_ham_clean <- tm_map(train_ham_Corpus, content_transformer(tolower))

#remove numbers
corpus_train_clean <- tm_map(corpus_train_clean, removeNumbers)
corpus_test_clean <- tm_map(corpus_test_clean, removeNumbers)
corpus_train_spam_clean <- tm_map(corpus_train_spam_clean, removeNumbers)
corpus_train_ham_clean <- tm_map(corpus_train_ham_clean, removeNumbers)

#remove stopwords
corpus_train_clean <- tm_map(corpus_train_clean, removeWords, stopwords())
corpus_test_clean <- tm_map(corpus_test_clean, removeWords, stopwords())
corpus_train_spam_clean <- tm_map(corpus_train_spam_clean, removeWords, stopwords())
corpus_train_ham_clean <- tm_map(corpus_train_ham_clean, removeWords, stopwords())

replacePunctuation <- function(x) {
  gsub("[[:punct:]]+", " ",x)
}

#replace punctuations
corpus_train_clean <- tm_map(corpus_train_clean, replacePunctuation)
corpus_test_clean <- tm_map(corpus_test_clean, replacePunctuation)
corpus_train_spam_clean <- tm_map(corpus_train_spam_clean, replacePunctuation)
corpus_train_ham_clean <- tm_map(corpus_train_ham_clean, replacePunctuation)

#implement stem
corpus_train_clean <- tm_map(corpus_train_clean, stemDocument)
corpus_test_clean <- tm_map(corpus_test_clean, stemDocument)
corpus_train_spam_clean <- tm_map(corpus_train_spam_clean, stemDocument)
corpus_train_ham_clean <- tm_map(corpus_train_ham_clean, stemDocument)

#remove white spaces
corpus_train_clean <- tm_map(corpus_train_clean, stripWhitespace)
corpus_test_clean <- tm_map(corpus_test_clean, stripWhitespace)
corpus_train_spam_clean <- tm_map(corpus_train_spam_clean, stripWhitespace)
corpus_train_ham_clean <- tm_map(corpus_train_ham_clean, stripWhitespace)

#convert to plain document
corpus_train_clean <- tm_map(corpus_train_clean, PlainTextDocument)
corpus_test_clean <- tm_map(corpus_test_clean, PlainTextDocument)
corpus_train_spam_clean <- tm_map(corpus_train_spam_clean, PlainTextDocument)
corpus_train_ham_clean <- tm_map(corpus_train_ham_clean, PlainTextDocument)


#create DocumentTermMatrix
corpus_train_dtm <- DocumentTermMatrix(corpus_train_clean)
corpus_test_dtm <- DocumentTermMatrix(corpus_test_clean)

install.packages("e1071")
library(e1071)

#writing a function to change the data to categorical format
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

#apply the function
train_final <- apply(corpus_train_dtm, MARGIN = 2, convert_counts)
test_final <- apply(corpus_test_dtm, MARGIN = 2, convert_counts)

#creating train labels
train_spam_val<- rep("SPAM",length(1:123))
train_ham_val<- rep("HAM",length(1:340))
train_labels <-as.factor(c(train_spam_val,train_ham_val))

#creating test labels
test_spam_val<- rep("SPAM",length(1:130))
test_ham_val<- rep("HAM",length(1:348))
test_labels <-as.factor(c(test_spam_val,test_ham_val))

#Implement Naive Bayes classification
classifier1 <- naiveBayes(train_final, train_labels, laplace = 1)
test_pred <- predict(classifier1, test_final)

library(gmodels)
CrossTable(test_pred, test_labels, prop.chisq = FALSE, prop.t = FALSE, dnn = c('predicted', 'actual'))
