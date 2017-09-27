grading <- function(standard, submission, tol=1e-2){
	studentName <- paste(submission$firstName, submission$lastName)
	fixedTitle <- c("firstName", "lastName", "GatorLinkID")
	titles <- setdiff(names(standard), fixedTitle)
	scoreTitle <- rep(6,length(titles))
	scores <- numeric(length(titles))
	names(scores) <- titles
	names(scoreTitle) <- titles
	endW <- sapply(titles,function(x) substring(x,nchar(x), nchar(x)))
	scoreTitle[ endW == "a" | endW == "b"] <- 3
	
	scores["A1"] <- abs(submission$A1 - standard$A1) < tol * standard$A1
	scores["A2"] <- all(submission$A2 %in% standard$A2) & all(standard$A2 %in% submission$A2)
	scores["A3"] <- all(submission$A3 %in% standard$A3) & all(standard$A3 %in% submission$A3)
	scores["A4"] <- all(submission$A4 %in% standard$A4) & all(standard$A4 %in% submission$A4)
	scores["B1"] <- abs(submission$B1 - standard$B1) < tol * standard$B1
	scores["B2"] <- abs(submission$B2 - standard$B2) < tol * standard$B2
	if(!is.numeric(submission$B3)){
	  scores["B3"] <- 0 
	} else {
	  scores["B3"] <- abs(submission$B3 - standard$B3) < tol * standard$B3
	}
	 
	scores["B4"] <- 1
	scores["B5"] <- 1
	
	scores["C1a"] <- abs(submission$C1a - standard$C1a) < tol
	scores["C1b"] <- abs(submission$C1b - standard$C1b) < tol
	scores["C2"] <- all(submission$C2 %in% standard$C2) & all(standard$C2 %in% submission$C2)
	scores["C3a"] <- all(submission$C3a %in% standard$C3a) & all(standard$C3a %in% submission$C3a)
	scores["C3b"] <- all(submission$C3b %in% standard$C3b) & all(standard$C3b %in% submission$C3b)
	scores["C4"] <- all(submission$C4 %in% standard$C4) & all(standard$C4 %in% submission$C4)
	scores["C5a"] <- abs(unlist(submission$C5a) - standard$C5a) < tol * standard$C5a
	scores["C5b"] <- abs(unlist(submission$C5b) - standard$C5b) < tol * standard$C5b
	scores["C6"] <- abs(submission$C6 - standard$C6) < tol * standard$C6
	
	gradeInfo <- list(name=studentName, scores=scores, grade=sum(scores * scoreTitle) + 10)
	return(gradeInfo)
}

