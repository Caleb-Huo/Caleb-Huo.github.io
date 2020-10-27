#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' author: "Zhiguang Huo (Caleb)"
#' date: "Wednesday October 11, 2017"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: Linear Mixed Model
#' ---
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
#' 
#' From linear model to linear mixed model
#' ===
#' 
#' Linear model:
#' 
#' - subjects are independent.
#' - all the varibility are explained by the independent variable and error term.
#' 
#' $$Y = X^\top\beta + \varepsilon$$
#' 
#' Linear mixed model: 
#' 
#' - account for variation that is not explained by the independent variables.
#' - account for correlated structure between subjects.
#' 
#' voice pitch data
#' ===
#' 
## -----------------------------------------------------------------------------
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
## -----------------------------------------------------------------------------
head(d)

with(d, table(subject, gender))

which(apply(d,1,function(x) any(is.na(x))))

#' 
#' visualize the data
#' ===
#' 
## -----------------------------------------------------------------------------
library(ggplot2)
ggplot(aes(attitude, frequency, color = subject), data=d) + facet_grid(~subject) + geom_boxplot() + geom_jitter() + theme_bw()

#' 
#' Fit linear model
#' ===
#' 
#' - Outcome variable: frequency
#' - primary predictor: attitude
#' - important covariate varaible: gender
#' - $\varepsilon$: deviations from our predictions due to random factors that we cannot control experimentally
#' 
#' $$pitch \sim politeness + gender + \varepsilon$$
#' 
#' Linear model in R
#' ===
#' 
## -----------------------------------------------------------------------------
lmFit <- lm(frequency ~ attitude + gender, data = d)
summary(lmFit)

#' 
#' Note: 1 observation deleted due to missingness
#' 
#' Problem with linear model
#' ===
#' 
#' - independence: since each subject gave multiple responses, multiple responses from the same subject cannot be regarded as independent from each other.
#' - under the same scenario, people should also react similarly, not totally independent.
#' 
#' data independence
#' ===
#' 
## -----------------------------------------------------------------------------
pol_subj = subset(d, attitude == "pol")
inf_subj = subset(d, attitude == "inf")
pol_subj$attitude <- NULL
inf_subj$attitude <- NULL
names(pol_subj)[names(pol_subj) == "frequency"] <- "frequency_pol"
names(inf_subj)[names(inf_subj) == "frequency"] <- "frequency_inf"

subj_merge <- merge(pol_subj, inf_subj)
head(subj_merge)

#' 
#' Subject dependence
#' ===
#' 
#' 
## -----------------------------------------------------------------------------
ggplot(data=subj_merge, aes(frequency_pol, frequency_inf)) + 
  geom_point(aes(col=subject)) + 
  geom_smooth(method="lm")

#' 
#' Subject dependence
#' ===
#' 
## -----------------------------------------------------------------------------
ggplot(data=subj_merge, aes(frequency_pol, frequency_inf)) + 
  geom_point(aes(col=subject)) + 
  stat_ellipse(aes(col=subject)) +
  geom_smooth(method="lm")

#' 
#' 
#' scenario independence
#' ===
#' 
#' 
## -----------------------------------------------------------------------------
ggplot(data=subj_merge, aes(frequency_pol, frequency_inf)) + 
  geom_point(aes(col=scenario0)) + 
  geom_smooth(method="lm")

#' 
#' scenario independence
#' ===
#' 
#' 
## -----------------------------------------------------------------------------
ggplot(data=subj_merge, aes(frequency_pol, frequency_inf)) + 
  geom_point(aes(col=scenario0)) + 
  stat_ellipse(aes(col=scenario0)) +
  geom_smooth(method="lm")

#' 
#' 
#' Mixed model
#' ===
#' 
#' - Linear model is "fixed-effects only" model
#'     - one or more fixed effects
#'     - a general error term $\varepsilon$
#'     - error term is considered no structure.
#' 
#' - Mixed model is a mixture of fixed effect and random effect
#'     - Fixed effects: the independent variables that are normally included in your analyses. For instance: gender, age, your study conditions
#'     - Random effects: the variables that are specific to your data sample. (E.g Subject)
#'     
#' Random effect
#' ===
#' - add a random effect for subject
#' - allows us to resolve this non-independence data structure for each subject
#' - allows missing data
#' 
#' Random Intercepts
#' ===
#' 
#' - linear model
#' $$pitch \sim politeness + gender + \varepsilon$$
#' 
#' - linear model with random effect (random intercept)
#' $$pitch \sim politeness + gender + (1|subject) + \varepsilon$$
#' $(1|subject)$ is the R syntax for a random intercept.
#' 
#' - intercepts are different for each subject
#' - "1" stands for the intercept in R
#' - this formula indicates multiple responses per subject
#' - resolves the non-independence multiple responses by the same subject
#' - the formula still contains a general error term $\varepsilon$
#' 
#' 
#' Fit linear mixed model in R
#' ===
#' 
#' - In R, lmer() function in lme4 package will do the job
#' 
## -----------------------------------------------------------------------------
lm(frequency ~ attitude, data = d)
library(lme4)
library(lmerTest) ## this package helps provide a p-value
tryCatch({lmer(frequency ~ attitude, data = d)}, error=function(e) print(e))

#' 
#' - You have to specify random effect
#' 
#' Fit linear mixed model in R
#' ===
## -----------------------------------------------------------------------------
rs_subj_reml = lmer(frequency ~ attitude + (1 | subject), data = d)
summary(rs_subj_reml)

#' 
#' Further explore rs_subj_reml
#' ===
#' 
## -----------------------------------------------------------------------------
coef(rs_subj_reml)
AIC(rs_subj_reml)

#' 
#' How to write the linear mixed model mathmatically
#' ===
#' 
#' $$Y_{ijk} = \mu + \beta_j + \alpha_i + \varepsilon_{ijk}$$
#' 
#' - $i$: subject index
#' - $j$: attitude index
#' - $k$: scenario index
#' - $\mu$: intercept
#' - $\beta_j$: attitude coefficient of $j^{th}$ group
#' - $\alpha_i \sim N(0, \sigma^2_\alpha)$: random intercept for the $i^{th}$ subject
#' - $\varepsilon_{ijk} \sim N(0, \sigma^2)$: 
#' 
#' 
#' Fit linear mixed model in R (REML = FALSE)
#' ===
#' 
## -----------------------------------------------------------------------------
rs_subj_ml = lmer(frequency ~ attitude + (1 | subject), REML = FALSE, data = d)
summary(rs_subj_ml)

#' 
#' REML represents restricted maximum likelihood
#' 
#' - if REML=FALSE, maximum likelihood estimation will be used
#' - its estimation does not base estimates on a maximum likelihood fit of all the information
#' - but instead uses a likelihood function calculated from a transformed set of data
#' - nuisance parameters have no effect.
#' 
#' 
#' Compare to linear model
#' ===
## -----------------------------------------------------------------------------
summary(lm(frequency ~ attitude, data = d))

#' 
#' Without adjusting for random effect, attitude won't be a significant predictor.
#' 
#' 
#' Getting p-values
#' ===
#' 
## -----------------------------------------------------------------------------
rs_subj_reml = lmer(frequency ~ attitude + (1 | subject), data = d) 
anova(rs_subj_reml, ddf = "Kenward-Roger")

## likelihood ratio test: comparing full model and reduced model
## get the p-value for reduced variable
## have to use REML = FALSE
rs_subj_ml = lmer(frequency ~ attitude + (1 | subject), REML = FALSE, data = d)
rs_null_ml = lmer(frequency ~ 1 + (1 | subject), REML = FALSE, data = d)

anova(rs_null_ml, rs_subj_ml)

#' 
#' add another covariate gender and its interaction with attitude
#' ===
## -----------------------------------------------------------------------------
summary(lmer(frequency ~ attitude * gender +  (1 | subject), data = d))

#' 
#' Investigate scenario
#' ===
#' 
## -----------------------------------------------------------------------------
ggplot(aes(1, frequency, col=scenario0), data=d) + 
  geom_boxplot() + 
  geom_jitter() + 
  facet_grid(attitude ~ scenario0)  + 
  theme_bw()

#' 
#' 
#' A larger mixed model
#' ===
#' 
## -----------------------------------------------------------------------------
rs_intergen_subjscene_ml = lmer(frequency ~ attitude + gender + (1 | subject) + 
    (1 | scenario), REML = FALSE, data = d)
summary(rs_intergen_subjscene_ml)

#' 
#' A larger mixed model
#' ===
## -----------------------------------------------------------------------------
coef(rs_intergen_subjscene_ml)

#' 
#' Random Slopes
#' ===
#' 
## -----------------------------------------------------------------------------
ggplot(aes(attitude, frequency, color = subject), data=d) + 
  geom_point() + 
  geom_smooth(method="lm", aes(group=1)) + 
  facet_wrap(~subject ) + 
  theme_bw()

#' 
#' Random Slopes
#' ===
#' 
## -----------------------------------------------------------------------------
ggplot(data=subj_merge, aes(frequency_pol, frequency_inf, col=scenario0)) + 
  geom_point() + 
  geom_smooth(method="lm", se = F)

#' 
#' 
#' Random Slopes in R
#' ===
#' 
## -----------------------------------------------------------------------------
politeness.model.rs = lmer(frequency ~  attitude + gender + (1 + attitude | subject) + 
    (1 | scenario), REML = FALSE, data = d)
summary(politeness.model.rs)

#' 
#' Random Slopes in R
#' ===
## -----------------------------------------------------------------------------
coef(politeness.model.rs)

#' 
#' Is random slope necessary
#' ===
## -----------------------------------------------------------------------------
rs_gen_subjscene_con_reml = lmer(frequency ~ attitude + gender + (1 + attitude | 
    subject) + (1 | scenario), REML = TRUE, data = d)

rs_gen_subjscene_reml = lmer(frequency ~ attitude + gender + (1 | subject) + (1 | 
    scenario), REML = TRUE, data = d)

anova(rs_gen_subjscene_reml, rs_gen_subjscene_con_reml)

#' 
#' - it appears that we don't need to include random slope for condition in the model 
#' - however, others would argue that we should keep our models maximal
#' 
#' 
#' Beyond linear mixed model
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
#' Another example sleep study
#' ===
#' 
## -----------------------------------------------------------------------------
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
#' Simple longitudinal data
#' ===
#' 
#' - Repeated measures data consist of measurements of a response (and, perhaps, some covariates) on several experimental (or observational) units.
#' - Longitudinal data are repeated measures data in which the observations are taken over time.
#' - We wish to characterize the response over time within subjects and the variation in the time trends between subjects.
#' 
#' Data exploratory
#' ===
#' 
## -----------------------------------------------------------------------------
ggplot(aes(Days, Reaction, col=Subject), data = sleepStudy) + geom_point() + geom_smooth(method="lm") + facet_wrap(~Subject)

#' 
#' 
#' Fit linear mixed model in R
#' ===
## -----------------------------------------------------------------------------
fm1 <- lmer(Reaction ~ Days + (Days|Subject), sleepstudy)
summary(fm1)

#' 
#' coef
#' ===
## -----------------------------------------------------------------------------
coef(fm1)

#' 
#' generate .R file
#' ===
#' 
## -----------------------------------------------------------------------------
knitr::purl("mixedModel.rmd", output = "mixedModel.R ", documentation = 2)

