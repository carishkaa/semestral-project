# semestral-project
RNASeq data classifier

Background: Kidney transplant recipients with “operational tolerance” (OT) maintain a functioning graft without immunosuppressive (IS) drugs, thus avoiding treatment complications. 

Problems: 
- IS drugs can influence gene-expression signatures aiming to identify OT among treated KTRs.
- Small number of samples instead of enormous number of genes.

What have I done for now:
Simple classifier based on Logistic Regression. 
For a more accurate classification, we get rid of "unnecessary" genes by using feature selection methods, which was performed within the 20 x 5-fold cross validation.

In future:
- generate artificial data
- add IS as features
- maybe try some other classifiers (SVM, Adaboost)
- confounding control...
