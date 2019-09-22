args = commandArgs(trailingOnly = TRUE) ## pass in external argument

rowID <- args[1]
aarg <- as.numeric(rowID)

WD <- "/ufrc/phc6068/share/zhuo/example/testR2" ## change to your own directory
dir.create(WD, re=T) ## force to create this folder

setwd(WD) ## set to your own directory!

mycars <- mtcars[aarg,]
filename <- paste0("arg",aarg,".csv")
write.csv(mycars,filename)
