#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' author: "Zhiguang Huo (Caleb)"
#' date: "Wednesday October 27, 2021"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: Linear Mixed Model
#' ---
#' 
#' 
#' 
#' From linear model to linear mixed model
#' ===
#' 
#' - Linear model:
#'     - Subjects are independent.
#'     - Cannot handle missing date (missing in Y).
#' 
#' $$Y = X^\top\beta + \varepsilon$$
#' 
#' - Linear mixed model: 
#'     - Account for correlated structure between subjects.
#'     - Can accommodate missing data (missing in Y).
#' 
#' voice pitch data
#' ===
#' 
## -------------------------------------------------------------
d = read.csv("https://caleb-huo.github.io/teaching/data/politeness/politeness_data.csv")
str(d)
d$attitude <- as.factor(d$attitude)
d$scenario0 <- as.factor(d$scenario)

#' 
#' - subject: (6 subjects)
#' - gender: (F/M)
#' - scenario: different scenario (7 categories)
#' - attitude: 
#'     - inf: informal situations (e.g., explaining to a friend why you're late)
#'     - pol: politeness (e.g., giving an excuse for being late to a professor) 
#' - frequency: outcome variable, frequency of voice pitch
#' 
#' voice pitch data
#' ===
#' 
## -------------------------------------------------------------
head(d)

with(d, table(subject, gender))

which(apply(d,1,function(x) any(is.na(x)))) ## one subject contails missing data (missing outcome variable)

#' 
#' visualize the data
#' ===
#' 
## -------------------------------------------------------------
library(tidyverse)
ggplot(aes(attitude, frequency, color = subject), data=d) + facet_grid(~subject) + geom_boxplot() + geom_jitter() + theme_bw()

#' 
#' Fit linear model
#' ===
#' 
#' - Outcome variable: pitch (frequency)
#' - primary predictor: attitude (politeness)
#' - $\varepsilon$: deviations from our predictions due to random factors that we cannot control experimentally
#' 
#' $$frequency \sim \mu + politeness + \varepsilon$$
#' 
#' Linear model in R
#' ===
#' 
## -------------------------------------------------------------
lmFit <- lm(frequency ~ attitude, data = d)
summary(lmFit)

#' 
#' Note: 
#' 
#' - $1$ observation deleted due to missingness
#' - attitude is not a significant predictor
#' 
#' 
#' Problem with linear model
#' ===
#' 
#' $$frequency_{ijk} \sim \mu + politeness_{k} + \varepsilon_{ijk},$$
#' where $i$ is a subject index, $j$ is a scenario index, and $k$ is the politeness index.
#' 
#' Dependent structure: 
#' 
#' - Multiple responses from the same subject $i$ cannot be regarded as independent from each other.
#'     - $frequency_{i1k}, frequency_{i2k}, \ldots, frequency_{iJk}$ should be correlated
#' 
#' data independence
#' ===
#' 
## -------------------------------------------------------------
subj_spread <- d %>% spread(attitude, frequency)
subj_spread$scenario0 <- as.factor(subj_spread$scenario)
head(subj_spread)

#' 
#' Subject dependence
#' ===
#' 
## -------------------------------------------------------------
ggplot(data=subj_spread, aes(pol, inf)) + theme_bw() +
  geom_point(aes(col=subject)) + 
  stat_ellipse(aes(col=subject)) +
  geom_smooth(method="lm")

#' 
#' 
#' 
#' Random effects:
#' ===
#' 
#' - Fixed effects: 
#'     - Categorical variables with well-defined categories (e.g., gender, race)
#' - Random effects: 
#'     - We only sample a subset of the entire population
#'       - The 6 participants in the data are a subset of the entire population
#'       - Each participant has repeated measurement.
#'       - The “subject” effect $\alpha_i$ for subject $i$ is best thought of as random: $\alpha_i \sim N(0, \sigma_0^2)$
#' 
#' When to use random effects?
#' 
#' - A "subject" effect is random if we can think of the levels we observe for that subject to be samples from a larger population.
#' - Example: we are investigating the GPA of UF undergraduate students by surveying a small group of students.
#'     - Gender and race are considered fixed effect since their categories are well-defined.
#'     - Department is considered as random effect, since our survey may not covering all departments in UF.
#' 
#' Mixed model
#' ===
#' 
#'     
#' - Linear model is "fixed-effects only" model
#'     - one or more fixed effects
#'     - a general error term $\varepsilon$
#'     - error term is considered no structure.
#' 
#' - Mixed model is a mixture of fixed effect and random effect
#'     
#' 
#'     
#' Benefit of random effect models
#' ===
#' 
#' - Allows us to resolve this non-independence data structure for each subject
#' - Allows missing data (missing outcome variable)
#' 
#' 
#' Fit linear mixed model in R
#' ===
#' 
#' - In R, lmer() function in lme4 package will do the job
#' 
## -------------------------------------------------------------
alm <- lm(frequency ~ attitude, data = d)
suppressMessages(library(lme4))
suppressMessages(library(lmerTest)) ## this package helps provide a p-value

#' 
#' ```
#' lmer(frequency ~ attitude, data = d)
#' ```
#' - We have to specify random effect when using lmer funciton.
#' 
#' Fit linear mixed model in R
#' ===
## -------------------------------------------------------------
rs_subj_reml = lmer(frequency ~ attitude + (1 | subject), data = d)
summary(rs_subj_reml)

#' 
#' How to write the linear mixed model mathmatically
#' ===
#' 
#' $$Y_{ijk} = \mu + \gamma \mathbb{I}(k=pol) + \alpha_i + \varepsilon_{ijk}$$
#' 
#' - $i$: subject index
#' - $j$: scenario index
#' - $k$: attitude index
#' - $\mu$: intercept
#' - $\gamma$: attitude coefficient of the polite group
#' - $\alpha_i \sim N(0, \sigma^2_\alpha)$: random effect for the $i^{th}$ subject
#' - $\varepsilon_{ijk} \sim N(0, \sigma^2)$: 
#' - Total error term: $\alpha_i + \varepsilon_{ijk}$
#'   - so the error for $i1k$ and $i2k$ are dependent
#' 
#' - If there is no $\alpha_i$, the above model will reduce to linear regression model.
#' 
#' Further explore rs_subj_reml
#' ===
#' 
## -------------------------------------------------------------
coef(alm)
coef(rs_subj_reml)
AIC(rs_subj_reml)

#' 
#' Further explore rs_subj_reml
#' ===
#' 
#' - Subject total intercept:
#' $$\mu + \alpha_i$$
#'   - vary among different subjects
#' - Attitude effect
#' $$\beta_j$$
#'   - same among different subjects
#' 
#' 
#' Random Intercepts
#' ===
#' 
#' - linear model
#' $$frequency \sim 1 + politeness + \varepsilon$$
#' 
#' - linear model with random effect (random intercept)
#' $$frequency \sim 1 +  politeness +(1|subject) + \varepsilon$$
#' $(1|subject)$ is the R syntax for a random intercept.
#' 
#' - intercepts are different for each subject
#' - "$1$" stands for the intercept in R
#' - this formula indicates multiple responses per subject
#' - resolves the non-independence multiple responses by the same subject
#' - the formula still contains a general error term $\varepsilon$
#' 
#' 
#' 
#' Visualization for the random intercept
#' ===
#' 
## ----echo = FALSE---------------------------------------------
d2 <- d %>% mutate(attitude2 = ifelse(attitude=="inf", 0, 1))
rs_subj_reml = lmer(frequency ~ attitude2 + (1 | subject), data = d2)
acoef <-  coef(rs_subj_reml)$subject
data_lines <- NULL
for(i in 1:nrow(acoef)){
  adata <- tibble(attitude2 = c(0,1), frequency = c(acoef[i,1], acoef[i,1] + acoef[i,2]), subject=rownames(acoef)[i])
  data_lines <- rbind(data_lines, adata)
}

ggplot(aes(attitude2, frequency, color = subject), data=d2) + 
  scale_x_continuous(breaks=c(0,1),
        labels=c("inf", "pol")) + 
  geom_point() + 
  geom_line(data = data_lines, aes(attitude2, frequency, color = subject)) + 
  facet_wrap(~subject) + 
  theme_bw()
    

#' 
#' 
#' Fit linear mixed model in R (REML = FALSE)
#' ===
#' 
## -------------------------------------------------------------
rs_subj_ml = lmer(frequency ~ attitude + (1 | subject), REML = FALSE, data = d)
summary(rs_subj_ml)

#' 
#' REML option
#' ===
#' 
#' REML represents restricted maximum likelihood
#' 
#' - if REML=FALSE, maximum likelihood estimation will be used (biased estimation)
#' - if REML=TRUE, its estimation does not base on a maximum likelihood, but instead uses a likelihood function calculated from a transformed set of data (unbiased estimation)
#' 
#' 
#' 
#' Compare to linear model
#' ===
## -------------------------------------------------------------
summary(lm(frequency ~ attitude, data = d))

#' 
#' Without adjusting for random effect, attitude won't be a significant predictor.
#' 
#' 
#' Alternative ways to get p-values (likelihood ratio test)
#' ===
#' 
#' - likelihood ratio test: comparing full model and reduced model
#' - get the p-value for reduced variable
#' - need to use REML = FALSE, otherwise, anova will re-fit the reml model such that REML = FALSE.
#' 
## -------------------------------------------------------------
rs_subj_ml = lmer(frequency ~ attitude + (1 | subject), REML = FALSE, data = d)
rs_null_ml = lmer(frequency ~ 1 + (1 | subject), REML = FALSE, data = d)

anova(rs_null_ml, rs_subj_ml)

#' 
#' Add covariates
#' ===
## -------------------------------------------------------------
summary(lmer(frequency ~ attitude + gender +  (1 | subject), data = d))

#' 
#' 
#' Randon intercept and random slope
#' ===
#' 
#' - Random intercept only:
#' 
#' $$Y_{ijk} = (\mu + \alpha_i) + \gamma \mathbb{I}(k=pol) + \varepsilon_{ijk}$$
#' 
#' - Random intercept and random slope:
#' 
#' $$Y_{ijk} = (\mu + \alpha_i) + (\gamma + \eta_i) \mathbb{I}(k=pol) + \varepsilon_{ijk}$$
#' 
#' 
#' - $(\alpha_i, \eta_i)^\top \sim N(0, \Sigma)$: random intercept and random slope for the $i^{th}$ subject
#' - $\varepsilon_{ijk} \sim N(0, \sigma^2)$: 
#' 
#' 
#' 
#' Visualization for the random slope
#' ===
#' 
## ----echo = FALSE---------------------------------------------
d2 <- d %>% mutate(attitude2 = ifelse(attitude=="inf", 0, 1))
rs_subj_reml = lmer(frequency ~ attitude2 + (attitude2 | subject), data = d2)
acoef <-  coef(rs_subj_reml)$subject
data_lines <- NULL
for(i in 1:nrow(acoef)){
  adata <- tibble(attitude2 = c(0,1), frequency = c(acoef[i,1], acoef[i,1] + acoef[i,2]), subject=rownames(acoef)[i])
  data_lines <- rbind(data_lines, adata)
}

ggplot(aes(attitude2, frequency, color = subject), data=d2) + 
  scale_x_continuous(breaks=c(0,1),
        labels=c("inf", "pol")) + 
  geom_point() + 
  geom_line(data = data_lines, aes(attitude2, frequency, color = subject)) + 
  facet_wrap(~subject) + 
  theme_bw()
    

#' 
#' 
#' Random Slopes in R
#' ===
#' 
## -------------------------------------------------------------
politeness.model.rs = lmer(frequency ~  attitude + gender + (1 + attitude | subject), REML = FALSE, data = d)
summary(politeness.model.rs)

#' 
#' Random Slopes in R
#' ===
## -------------------------------------------------------------
coef(politeness.model.rs)

#' 
#' Is random slope necessary
#' ===
## -------------------------------------------------------------
rs_gen_subjscene_con_reml = lmer(frequency ~ attitude + gender + (1 + attitude | 
    subject), REML = TRUE, data = d)

rs_gen_subjscene_reml = lmer(frequency ~ attitude + gender + (1 | subject), REML = TRUE, data = d)

anova(rs_gen_subjscene_reml, rs_gen_subjscene_con_reml)

#' 
#' - it appears that we don't need to include random slope for condition in the model, because the likelihood ratio test is not significant
#' - however, others would argue that we should keep our models maximal
#' 
#' 
#' Beyond linear mixed model (optional)
#' ===
#' 
#' If the outcome variable is
#' 
#' - Continuous: use a linear regression model with mixed effects
#' - Binary: use a logistic regression model with mixed effects
#' - Count: use a Poisson/negative binomial regression model with mixed effects
#' 
#' Function lmer is used to fit linear mixed models, function glmer is used to fit generalized (non-Gaussian) linear mixed models.
#' 
#' 
#' Longitudinal data
#' ===
#' 
#' - Longitudinal data are repeated measures data in which the observations are taken over time.
#' - We wish to characterize the response over time within subjects and the variation in the time trends between subjects.
#' 
#' Longitudinal data -- sleep study
#' ===
#' 
## -------------------------------------------------------------
sleepStudy <- read.table("https://caleb-huo.github.io/teaching/data/sleep/sleepstudy.txt", header=T, row.names = 1)
sleepStudy$Subject <- as.factor(sleepStudy$Subject)

head(sleepStudy, n=10)
str(sleepStudy)

#' 
#' 
#' 
#' Sleep deprivation data
#' ===
#' 
#' - This laboratory experiment measured the effect of sleep deprivation on cognitive performance. 
#' - There were 18 subjects, chosen from the population of interest (long-distance truck drivers), in the 10 day trial. These subjects were restricted to 3 hours sleep per night during the trial.
#' - On each day of the trial each subject’s reaction time was measured. The reaction time shown here is the average of several measurements.
#' - These data are balanced in that each subject is measured the same number of times and on the same occasions.
#' 
#' 
#' Data exploratory
#' ===
#' 
## -------------------------------------------------------------
ggplot(aes(Days, Reaction, col=Subject), data = sleepStudy) + geom_point() + geom_smooth(method="lm") + facet_wrap(~Subject)

#' 
#' 
#' Fit linear mixed model in R
#' ===
#' 
#' - Examine the relationship of Reaction time with respect to Days (slope)
#' - Each subject has repeated measurement of Reaction time
#' - Consider subject specific random intercept only
#' 
#' 
#' ---
#' 
## -------------------------------------------------------------
fm1 <- lmer(Reaction ~ Days + (1|Subject), sleepstudy)
summary(fm1)

#' 
#' 
#' Fit linear mixed model in R
#' ===
#' 
#' - Examine the relationship of Reaction time with respect to Days (slope)
#' - Each subject has repeated measurement of Reaction time
#' - Consider both subject specific random intercept and subject specific random slope
#' 
#' 
#' ---
#' 
## -------------------------------------------------------------
fm1 <- lmer(Reaction ~ Days + (1 + Days|Subject), sleepstudy)
#fm1 <- lmer(Reaction ~ Days + (Days|Subject), sleepstudy) ## same as this one
summary(fm1)

#' 
#' 
#' coef
#' ===
## -------------------------------------------------------------
coef(fm1)

#' 
#' 
#' References:
#' ===
#' 
#' Credits give to:
#' 
#' - <https://web.stanford.edu/class/psych252/section_2013/Mixed_models_tutorial.html>
#' - Winter, B. (2013). Linear models and linear mixed effects models in R with linguistic applications. arXiv:1308.5499. <https://arxiv.org/pdf/1308.5499.pdf>
#' - <http://lme4.r-forge.r-project.org/slides/2011-03-16-Amsterdam/2Longitudinal.pdf>
#' - <http://statweb.stanford.edu/~jtaylo/courses/stats203/notes/fixed+random.pdf>
