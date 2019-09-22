WD <- "/ufrc/phc6068/share/zhuo/example/testR" ## change to your own directory
dir.create(WD, re=T) ## force to create this folder

setwd(WD) ## set to your own directory!
mycars <- mtcars
write.csv(mycars,"mycars.csv")
