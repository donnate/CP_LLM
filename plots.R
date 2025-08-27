library(dplyr)
library(readr)   # or data.table::fread if faster
library(purrr)   # for map_dfr
library(tidyverse)
# list all your results files (adjust pattern and path if needed)
files <- list.files(
  "~/Documents/CP_LLM/results_new/",
  pattern = "^experiment_on_alpha*\\.csv$",
  full.names = TRUE
)
files
# read and combine them all
results_df <- files %>%
  map_dfr(read_csv)   # or map_dfr(fread) if using data.table

results_df["choice"] = "CV"


files2 <- list.files(
  "~/Documents/CP_LLM/results_new/",
  pattern = "^CP_YJ_SIC.*\\.csv$",
  full.names = TRUE
)
files2
# read and combine them all
results_df2 <- files2 %>%
  map_dfr(read_csv)   # or map_dfr(fread) if using data.table

results_df2["choice"] = "SIC"

results_df = rbind(results_df,
                   results_df2)
unique(results_df$alpha)
# summarise like before
summary_df <- results_df %>%
  group_by(V, K, delta,alpha, epsilon, T, lambda_obs, rho, alpha_cp,
           n_train_docs, n_calib_docs,
           choice) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE)) %>%
  ungroup()

unique(results_df$alpha)
unique(summary_df$K)
unique(summary_df$delta)
unique(summary_df$epsilon)
unique(summary_df$lambda_obs)
unique(summary_df$alpha_cp)
unique(summary_df$n_train_docs)
unique(summary_df$n_calib_docs)
unique(summary_df$rho)
unique(summary_df$T)
unique(summary_df$choice)
colnames(summary_df)

summary_df

ggplot(summary_df %>% filter( epsilon ==0.5,
                              lambda_obs == 0.001,
                              rho==2) %>%
         mutate(temp = paste0("T  = ", T),
                delta_lab = paste0("delta  = ", delta),
                n_train_docs_lab = paste0("n_train_docs = ", n_train_docs),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs)),
       aes(x= T,  corr_calib_oracle,  shape = n_train_docs_lab, colour =as.factor(K))) + 
  geom_point()+
  geom_line()+
  #geom_point(aes(x= T,  r2_calib_oracle, shape = n_train_docs_lab, colour ="correlation with A_oracle"))+
  #geom_line(aes(x= T,  r2_calib_oracle, colour ="correlation with A_oracle"))+
facet_grid(delta_lab~ n_calib_docs_lab )+
  ylab("Correlation of A_hat vs A") + 
  xlab("Temperature") +
  theme_bw()


colnames(summary_df)
ggplot(summary_df %>% filter(
                              n_calib_docs == 50,
                              K==10,
                              rho==2,
                              epsilon==0.5) %>%
         mutate(temp = paste0("T  = ", T),
                delta_lab = paste0("delta = ", delta),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  #geom_point(aes(x=epsilon, phi_l1_unaug_full, color= "Unaugmented")) +
  #geom_line(aes(x=epsilon, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=T, cosine_mean_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=T, cosine_mean_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=T + 0.1, cosine_mean_obs, color= "filtered:obs")) +
  geom_line(aes(x=T+ 0.1, cosine_mean_obs,color= "filtered:obs")) +
  geom_point(aes(x=T+ 0.2, cosine_mean_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=T+ 0.2, cosine_mean_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=T+ 0.25, cosine_mean_marg, color= "filtered:CP_marg")) +
  geom_line(aes(x=T+ 0.25, cosine_mean_marg,color= "filtered:CP_marg")) +
  geom_point(aes(x=T+ 0.4, cosine_mean_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=T+ 0.4, cosine_mean_cp,color= "filtered:CP_cond")) +
  facet_grid( lambda_obs_lab ~ delta_lab )+
  ylab("Diversity metric: cosine similarity of generation with input") + 
  xlab("Temperature") +
  theme_bw()



ggplot(summary_df %>% filter(# lambda_obs == 0.01,
                              K==10,
                              rho==2,
                              lambda_obs <0.1,
                              n_calib_docs == 50,
                              epsilon==0.5) %>%
         mutate(temp = paste0("T  = ", T),
                delta_lab = paste0("delta  = ", delta),
                n_train_docs_lab = paste0("n_train_docs = ", n_train_docs),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  #geom_point(aes(x=epsilon, phi_l1_unaug_full, color= "Unaugmented")) +
  #geom_line(aes(x=epsilon, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=delta, cosine_med_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=delta, cosine_med_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=delta, cosine_med_obs, color= "filtered:obs")) +
  geom_line(aes(x=delta, cosine_med_obs,color= "filtered:obs")) +
  geom_point(aes(x=delta, cosine_med_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=delta, cosine_med_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=delta, cosine_mean_marg, color= "filtered:CP_marg")) +
  geom_line(aes(x=delta, cosine_mean_marg,color= "filtered:CP_marg")) +
  geom_point(aes(x=delta, cosine_med_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=delta, cosine_med_cp,color= "filtered:CP_cond")) +
  facet_grid( temp ~ lambda_obs_lab )+
  ylab("Diversity metric: cosine similarity of generation with input") + 
  xlab("Delta") +
  theme_bw()


ggplot(summary_df %>% filter(# lambda_obs == 0.01,
  K==3,
  rho==2,
  n_calib_docs == 100,
  epsilon==0.5) %>%
    mutate(temp = paste0("T  = ", T),
           delta_lab = paste0("delta  = ", delta),
           n_train_docs_lab = paste0("n_train_docs = ", n_train_docs),
           n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
           lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  #geom_point(aes(x=epsilon, phi_l1_unaug_full, color= "Unaugmented")) +
  #geom_line(aes(x=epsilon, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=lambda_obs, cosine_med_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=lambda_obs, cosine_med_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=lambda_obs, cosine_med_obs, color= "filtered:obs")) +
  geom_line(aes(x=lambda_obs, cosine_med_obs,color= "filtered:obs")) +
  geom_point(aes(x=lambda_obs, cosine_med_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=lambda_obs, cosine_med_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=lambda_obs, cosine_mean_marg, color= "filtered:CP_marg")) +
  geom_line(aes(x=lambda_obs, cosine_mean_marg,color= "filtered:CP_marg")) +
  geom_point(aes(x=lambda_obs, cosine_med_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=lambda_obs, cosine_med_cp,color= "filtered:CP_cond")) +
  facet_grid( temp ~ delta_lab )+
  scale_y_log10()+
  scale_x_log10()+
  ylab("Diversity metric: cosine similarity of generation with input") + 
  xlab("lambda_obs") +
  theme_bw()


unique(summary_df$K)
ggplot(summary_df %>% filter( delta ==0.5, 
                              K==3,
                              rho==2,
                              n_calib_docs == 200) %>%
         mutate(temp = paste0("T  = ", T),
                epsilon_lab = paste0("epsilon = ", epsilon),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  #geom_point(aes(x=epsilon, phi_l1_unaug_full, color= "Unaugmented")) +
  #geom_line(aes(x=epsilon, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=lambda_obs, cosine_mean_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=lambda_obs, cosine_mean_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=lambda_obs, cosine_mean_obs, color= "filtered:obs")) +
  geom_line(aes(x=lambda_obs, cosine_mean_obs,color= "filtered:obs")) +
  geom_point(aes(x=lambda_obs, cosine_mean_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=lambda_obs, cosine_mean_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=lambda_obs, cosine_mean_marg, color= "filtered:CP_marg")) +
  geom_line(aes(x=lambda_obs, cosine_mean_marg,color= "filtered:CP_marg")) +
  facet_grid( temp ~ epsilon_lab )+
  ylab("Diversity metric: cosine similarity of generation with input") + 
  xlab("lambda_obs") +
  theme_bw()



ggplot(summary_df %>% filter( delta ==0.5, 
                              K==20,
                              rho==0,
                              lambda_obs == 0.005,
                              n_calib_docs == 200) %>%
         mutate(temp = paste0("T  = ", T),
                epsilon_lab = paste0("epsilon = ", epsilon),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  #geom_point(aes(x=epsilon, phi_l1_unaug_full, color= "Unaugmented")) +
  #geom_line(aes(x=epsilon, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=epsilon, cosine_mean_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=epsilon, cosine_mean_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=epsilon, cosine_mean_obs, color= "filtered:obs")) +
  geom_line(aes(x=epsilon, cosine_mean_obs,color= "filtered:obs")) +
  # geom_point(aes(x=epsilon, cosine_mean_cp, color= "filtered:CP_cond")) +
  # geom_line(aes(x=epsilon, cosine_mean_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=epsilon, cosine_mean_marg, color= "filtered:CP_marg")) +
  geom_line(aes(x=epsilon, cosine_mean_marg,color= "filtered:CP_marg")) +
  facet_grid( temp ~ . )+
  ylab("Diversity metric: cosine similarity of generation with input") + 
  xlab("lambda_obs") +
  theme_bw()





colnames(summary_df)

ggplot(summary_df %>% filter( epsilon ==0.5, 
                              K==3,
                              rho==2,
                              choice== "SIC",
                              n_calib_docs == 50,
                              lambda_obs <0.1) %>%
         mutate(temp = paste0("T  = ", T),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=delta, phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=delta, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=delta +0.1, phi_l1_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=delta +0.1, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=delta +0.2, phi_l1_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=delta +0.2, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=delta +0.3, phi_l1_aug_full_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=delta +0.3, phi_l1_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=delta, phi_l1_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=delta , phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=delta, phi_l1_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=delta, phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid( temp ~ lambda_obs_lab , scales = "free" )+
  ylab("|| A_hat - A ||_1") + 
  scale_y_log10()+
  #scale_x_log10()+
  xlab("delta") +
  theme_bw()

ggplot(summary_df %>% filter( epsilon ==0.5, 
                              K==20,
                              rho==2,
                              lambda_obs == 0.05,
                              n_calib_docs == 50,
                              lambda_obs <0.1) %>%
         mutate(temp = paste0("T  = ", T),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=delta, phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=delta, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=delta +0.1, phi_l1_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=delta +0.1, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=delta +0.2, phi_l1_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=delta +0.2, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=delta +0.3, phi_l1_aug_full_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=delta +0.3, phi_l1_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=delta, phi_l1_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=delta , phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=delta +0.3, phi_l1_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=delta +0.3, phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid( temp ~ choice , scales = "free" )+
  ylab("|| A_hat - A ||_1") + 
  scale_y_log10()+
  #scale_x_log10()+
  xlab("delta") +
  theme_bw()




ggplot(summary_df %>% filter( epsilon ==0.5, 
                              K==20,
                              rho==2,
                              lambda_obs == 0.05,
                              n_calib_docs == 50,
                              lambda_obs <0.1) %>%
         mutate(temp = paste0("T  = ", T),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  #geom_point(aes(x=delta, phi_l1_unaug_full, color= "Unaugmented")) +
  #geom_line(aes(x=delta, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=delta +0.1, n_added_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=delta +0.1, n_added_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=delta +0.2, n_added_obs, color= "filtered:obs")) +
  geom_line(aes(x=delta +0.2, n_added_obs,color= "filtered:obs")) +
  geom_point(aes(x=delta +0.3, n_added_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=delta +0.3, n_added_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=delta, n_added_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=delta , n_added_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=delta +0.3, n_added_marg, color= "filtered:CP_marg")) +
  geom_line(aes(x=delta +0.3, n_added_marg,color= "filtered:CP_marg")) +
  facet_grid( temp ~ choice , scales = "free" )+
  ylab("|| A_hat - A ||_1") + 
  scale_y_log10()+
  #scale_x_log10()+
  xlab("delta") +
  theme_bw()


colnames(summary_df)
ggplot(summary_df %>% filter( epsilon ==0.5, 
                              K==50,
                              rho==0,
                              lambda_obs <0.1,
                              T == 2) %>%
         mutate(temp = paste0("T  = ", T),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=delta, 0, color= "Unaugmented")) +
  geom_line(aes(x=delta, 0,color= "Unaugmented")) +
  geom_point(aes(x=delta +0.1, base_unf_accept_rate, color= "unfiltered")) +
  geom_line(aes(x=delta +0.1, base_unf_accept_rate,color= "unfiltered")) +
  geom_point(aes(x=delta +0.2, base_obs_accept_rate, color= "filtered:obs")) +
  geom_line(aes(x=delta +0.2, base_obs_accept_rate,color= "filtered:obs")) +
  geom_point(aes(x=delta +0.3, base_oracle_accept_rate, color= "filtered:oracle")) +
  geom_line(aes(x=delta +0.3, base_oracle_accept_rate,color= "filtered:oracle")) +
  geom_point(aes(x=delta, CP_accept_rate, color= "filtered:CP_cond")) +
  geom_line(aes(x=delta , CP_accept_rate,color= "filtered:CP_cond")) +
  geom_point(aes(x=delta, Marg_accept_rate, color= "filtered:CP_marg")) +
  geom_line(aes(x=delta, Marg_accept_rate,color= "filtered:CP_marg")) +
  facet_grid( n_calib_docs_lab ~ lambda_obs_lab , scales = "free" )+
  ylab("Accept rate") + 
  scale_y_log10()+
  #scale_x_log10()+
  xlab("delta") +
  theme_bw()


ggplot(summary_df %>% filter( delta ==0.5, 
                              K==20,
                              rho==2,
                              T == 2) %>%
         mutate(temp = paste0("T  = ", T),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=epsilon, topic_jsd_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=epsilon, topic_jsd_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=epsilon, topic_jsd_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=epsilon, topic_jsd_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=epsilon, topic_jsd_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=epsilon, topic_jsd_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=epsilon, topic_jsd_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=epsilon, topic_jsd_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=epsilon, topic_jsd_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=epsilon, topic_jsd_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid( lambda_obs_lab ~ n_calib_docs_lab )+
  ylab("Topic JSD") + 
  xlab("epsilon") +
  theme_bw()

unique(summary_df$delta)

ggplot(summary_df %>% filter(#rho == 0,
  rho ==2,
                             epsilon == 0.5,
  n_calib_docs ==50,
                             K==20) %>%
         mutate(temp = paste0("T  = ", T),
                delta_lab = paste0("delta = ", delta),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=T, phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=T, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=T, phi_l1_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=T, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=T, phi_l1_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=T, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=T, phi_l1_aug_full_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=T, phi_l1_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=T, phi_l1_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=T, phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=T, phi_l1_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=T, phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid( delta_lab ~ lambda_obs_lab , scales="free")+
  ylab("|| A_hat - A ||_1") + 
  scale_x_log10()+
  xlab("Temperature") +
  theme_bw()


ggplot(summary_df %>% filter(#rho == 0,
  rho ==2,
  epsilon == 0.5,
  choice== "SIC",
  lambda_obs ==0.01,
  n_calib_docs ==50) %>%
    mutate(temp = paste0("T  = ", T),
           delta_lab = paste0("delta = ", delta),
           n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
           lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=K, phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=K, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=K, phi_l1_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=K, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=K, phi_l1_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=K, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=K, phi_l1_aug_full_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=K, phi_l1_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=K, phi_l1_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=K, phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=K, phi_l1_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=K, phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid( delta_lab ~ temp , scales="free")+
  ylab("|| A_hat - A ||_1") + 
  scale_x_log10()+
  scale_y_log10()+
  xlab("K") +
  theme_bw()

ggplot(summary_df %>% filter(#rho == 0,
  rho ==2,
  epsilon == 0.5,
  delta==2,
  choice== "SIC",
  n_calib_docs ==50) %>%
    mutate(temp = paste0("T  = ", T),
           delta_lab = paste0("delta = ", delta),
           n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
           lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=lambda_obs, phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=lambda_obs, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=lambda_obs, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=lambda_obs, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=lambda_obs, phi_l1_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=lambda_obs, phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=lambda_obs, phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid( K ~ temp , scales="free")+
  ylab("|| A_hat - A ||_1") + 
  scale_x_log10()+
  scale_y_log10()+
  xlab("lambda_obs") +
  theme_bw()



ggplot(summary_df %>% filter(#rho == 0,
  #rho ==0,
  #n_calib_docs == 50,
  epsilon == 0.5,
  delta ==2, 
  K==20) %>%
    mutate(temp = paste0("T  = ", T),
           rho_lab = paste0("rho = ", rho),
           epsilon_lab = paste0("epsilon = ", epsilon),
           n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
           lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=T, phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=T, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=T, phi_l1_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=T, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=T, phi_l1_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=T, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=T, phi_l1_aug_full_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=T, phi_l1_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=T, phi_l1_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=T, phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=T, phi_l1_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=T, phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid( rho_lab + n_calib_docs ~ lambda_obs_lab , scales="free")+
  ylab("|| A_hat - A ||_1") + 
  scale_x_log10()+
  xlab("Temperature") +
  theme_bw()



colnames(summary_df)

ggplot(summary_df %>% filter(delta ==0.5,
                             K==20,
                             rho ==2,
                             lambda_obs ==0.05,
                             epsilon == 0.5) %>%
         mutate(temp = paste0("T  = ", T),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  #geom_point(aes(x=lambda_obs, phi_l1_unaug_full, color= "Unaugmented")) +
  #geom_line(aes(x=lambda_obs, phi_l1_unaug_full,color= "Unaugmented")) +
  # geom_point(aes(x=lambda_obs, base_unf_accept_rate, color= "unfiltered")) +
  # geom_line(aes(x=lambda_obs, base_unf_accept_rate,color= "unfiltered")) +
  geom_point(aes(x=T, base_obs_accept_rate, color= "filtered:obs")) +
  geom_line(aes(x=T, base_obs_accept_rate,color= "filtered:obs")) +
  geom_point(aes(x=T, CP_accept_rate, color= "filtered:CP_cond")) +
  geom_line(aes(x=T, CP_accept_rate,color= "filtered:CP_cond")) +
  geom_point(aes(x=T, Marg_accept_rate, color= "filtered:CP_marg")) +
  geom_line(aes(x=T, Marg_accept_rate,color= "filtered:CP_marg")) +
  facet_grid(n_calib_docs_lab ~ .)+
  ylab("Acceptance rate") + 
  #scale_y_log10()+
  xlab("Temperature") +
  theme_bw()


colnames(summary_df)
ggplot(summary_df %>% filter(rho == 2, delta ==2, 
                             K==20,
                             epsilon ==0.5) %>%
         mutate(temp = paste0("T  = ", T),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  #geom_point(aes(x=lambda_obs, phi_l1_unaug_full, color= "Unaugmented")) +
  #geom_line(aes(x=lambda_obs, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=lambda_obs, base_unf_miscoverage, color= "unfiltered")) +
  geom_line(aes(x=lambda_obs, base_unf_miscoverage,color= "unfiltered")) +
  # geom_point(aes(x=lambda_obs, base_obs_miscoverage, color= "filtered:obs")) +
  # geom_line(aes(x=lambda_obs, base_obs_miscoverage,color= "filtered:obs")) +
  geom_point(aes(x=lambda_obs, CP_miscoverage, color= "filtered:CP_cond")) +
  geom_line(aes(x=lambda_obs, CP_miscoverage,color= "filtered:CP_cond")) +
  geom_point(aes(x=lambda_obs, Marg_miscoverage, color= "filtered:CP_marg")) +
  geom_line(aes(x=lambda_obs, Marg_miscoverage,color= "filtered:CP_marg")) +
  geom_abline(aes(intercept = 0.1, slope=0), colour="black")+
  facet_grid( temp ~ n_calib_docs_lab )+
  ylab("Miscoverage rate") + 
  xlab("Lambda Obs") +
  theme_bw()


ggplot(summary_df %>% filter(n_calib_docs==200, delta ==0.5, 
                             K==20,
                             lambda_obs < 0.15,
                             epsilon ==0.5) %>%
         mutate(temp = paste0("T  = ", T),
                rho_lab = paste0("rho = ", rho),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=lambda_obs, phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=lambda_obs, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=lambda_obs, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=lambda_obs, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=lambda_obs, phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=lambda_obs, phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid( temp ~ rho_lab )+
  ylab("|| A_hat - A ||_1") + 
  xlab("Lambda Obs") +
  theme_bw()



colnames(summary_df)
ggplot(summary_df %>% filter(rho == 2,
                             K==20,
                             n_calib_docs == 50,
                             epsilon ==0.5) %>%
         mutate(temp = paste0("T  = ", T),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=delta, clf_acc_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=delta, clf_acc_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=delta, clf_acc_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=delta, clf_acc_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=delta, clf_acc_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=delta, clf_acc_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=delta, clf_acc_aug_full_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=delta, clf_acc_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=delta, clf_acc_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=delta, clf_acc_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=delta, clf_acc_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=delta, clf_acc_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid( temp ~ lambda_obs )+
  ylab("Classfication Accuracy") + 
  xlab("Delta") +
  theme_bw()

ggplot(summary_df %>% filter(rho == 2,
                             K==20,
                             n_calib_docs == 50,
                             epsilon ==0.5) %>%
         mutate(temp = paste0("T  = ", T),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=T, clf_acc_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=T, clf_acc_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=T, clf_acc_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=T, clf_acc_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=T, clf_acc_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=T, clf_acc_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=T, clf_acc_aug_full_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=T, clf_acc_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=T, clf_acc_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=T, clf_acc_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=T, clf_acc_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=T, clf_acc_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid( delta ~ lambda_obs )+
  ylab("Classfication Accuracy") + 
  xlab("T") +
  theme_bw()

colnames(summary_df)
ggplot(summary_df %>% filter(rho == 2,
                             delta==2,
                             n_calib_docs == 50,
                             epsilon ==0.5) %>%
         mutate(temp = paste0("T  = ", T),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=lambda_obs, reg_mse_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=lambda_obs, reg_mse_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=lambda_obs, reg_mse_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=lambda_obs, reg_mse_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=lambda_obs, reg_mse_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=lambda_obs, reg_mse_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=lambda_obs, reg_mse_aug_full_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=lambda_obs, reg_mse_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=lambda_obs, reg_mse_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=lambda_obs, reg_mse_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=lambda_obs, reg_mse_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=lambda_obs, reg_mse_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid( K ~temp , scales = "free" )+
  ylab("Classfication Accuracy") + 
  xlab("lambda_obs") +
  theme_bw()



:wq
colnames(summary_df)
ggplot(summary_df %>% filter(rho == 2, delta ==2, 
                             n_calib_docs ==50,
                             epsilon ==0.5) %>%
         mutate(temp = paste0("T  = ", T),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=lambda_obs, topic_jsd_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=lambda_obs, topic_jsd_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=lambda_obs, topic_jsd_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=lambda_obs, topic_jsd_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=lambda_obs, topic_jsd_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=lambda_obs, topic_jsd_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=lambda_obs, topic_jsd_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=lambda_obs, topic_jsd_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=lambda_obs, topic_jsd_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=lambda_obs, topic_jsd_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid( temp ~ K )+
  ylab("Topic JSD") + 
  xlab("Lambda Obs") +
  theme_bw()




ggplot(summary_df %>% filter(rho == 2, delta ==0.5, 
                             K==20,
                             epsilon ==0.5) %>%
         mutate(temp = paste0("T  = ", T),
                n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=lambda_obs, topic_cos_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=lambda_obs, topic_cos_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=lambda_obs, topic_cos_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=lambda_obs, topic_cos_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=lambda_obs, topic_cos_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=lambda_obs, topic_cos_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=lambda_obs, topic_cos_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=lambda_obs, topic_cos_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=lambda_obs, topic_cos_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=lambda_obs, topic_cos_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid( temp ~ n_calib_docs_lab )+
  ylab("Cosine") + 
  xlab("Lambda Obs") +
  theme_bw()



ggplot(summary_df %>% filter(rho == 10, delta ==0.5, 
                             n_calib_docs== 200) %>%
         mutate(temp = paste0("T  = ", T),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=T, phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=T, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=T, phi_l1_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=T, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=T, phi_l1_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=T, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=T, phi_l1_aug_full_cp, color= "filtered:CP")) +
  geom_line(aes(x=T, phi_l1_aug_full_cp,color= "filtered:CP")) +
  facet_grid( lambda_obs_lab ~ epsilon )+
  ylab("|| A_hat - A ||_1") + 
  xlab("Temperature") +
  scale_x_log10()+
  theme_bw()


ggplot(summary_df %>% filter(T == 2, delta ==0.5, 
                             n_calib_docs== 200) %>%
         mutate(temp = paste0("T  = ", T),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=rho, phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=rho, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=rho, phi_l1_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=rho, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=rho, phi_l1_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=rho, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=rho, phi_l1_aug_full_cp, color= "filtered:CP")) +
  geom_line(aes(x=rho, phi_l1_aug_full_cp,color= "filtered:CP")) +
  facet_grid( lambda_obs_lab ~ epsilon )+
  ylab("|| A_hat - A ||_1") + 
  xlab("rho") +
  #scale_x_log10()+
  theme_bw()




ggplot(summary_df %>% filter(rho == 0, delta ==0.5, lambda_obs ==0.01) %>%
         mutate(temp = paste0("T  = ", T),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=n_calib_docs, phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=n_calib_docs, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=n_calib_docs, phi_l1_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=n_calib_docs, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=n_calib_docs, phi_l1_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=n_calib_docs, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=n_calib_docs, phi_l1_aug_full_cp, color= "filtered:CP")) +
  geom_line(aes(x=n_calib_docs, phi_l1_aug_full_cp,color= "filtered:CP")) +
  facet_grid( temp ~ epsilon )+
  ylab("|| A_hat - A ||_1") + 
  xlab("Number of calib docs") +
  scale_x_log10()+
  theme_bw()

ggplot(summary_df %>%  filter(rho == 0, delta ==0.5, lambda_obs ==0.01) %>%
         mutate(temp = paste0("T  = ", T),
                rho_lab = paste0("rho = ", rho),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  #geom_point(aes(x=T, base_unf_accept_rate, color= "Unaugmented")) +
  #geom_line(aes(x=T, base_unf_accept_rate,color= "Unaugmented")) +
  geom_point(aes(x=n_calib_docs, base_unf_accept_rate, color= "unfiltered")) +
  geom_line(aes(x=n_calib_docs, base_unf_accept_rate,color= "unfiltered")) +
  geom_point(aes(x=n_calib_docs, base_obs_accept_rate, color= "filtered:obs")) +
  geom_line(aes(x=n_calib_docs, base_obs_accept_rate,color= "filtered:obs")) +
  geom_point(aes(x=n_calib_docs, CP_accept_rate, color= "filtered:CP")) +
  geom_line(aes(x=n_calib_docs, CP_accept_rate,color= "filtered:CP")) +
  facet_grid( temp ~ epsilon )+
  ylab("Acceptance rate") + 
  xlab("Number of Calib docs") +
  scale_x_log10()+
  theme_bw()




ggplot(summary_df %>% filter(delta ==0.5, epsilon == 0.5,
                             n_calib_docs ==200,
                             lambda_obs < 0.1) %>%
         mutate(temp = paste0("T  = ", T),
                rho_lab = paste0("rho = ", rho),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  #geom_point(aes(x=T, base_unf_accept_rate, color= "Unaugmented")) +
  #geom_line(aes(x=T, base_unf_accept_rate,color= "Unaugmented")) +
  geom_point(aes(x=T, base_unf_accept_rate, color= "unfiltered")) +
  geom_line(aes(x=T, base_unf_accept_rate,color= "unfiltered")) +
  geom_point(aes(x=T, base_obs_accept_rate, color= "filtered:obs")) +
  geom_line(aes(x=T, base_obs_accept_rate,color= "filtered:obs")) +
  geom_point(aes(x=T, CP_accept_rate, color= "filtered:CP")) +
  geom_line(aes(x=T, CP_accept_rate,color= "filtered:CP")) +
  facet_grid( lambda_obs_lab ~ rho_lab )+
  ylab("Acceptance rate") + 
  xlab("Temperature") +
  scale_x_log10()+
  theme_bw()

colnames(summary_df)


ggplot(summary_df %>% filter(delta ==0.5, epsilon == 0.5,
                             n_calib_docs ==200,
                             lambda_obs < 0.1) %>%
         mutate(temp = paste0("T  = ", T),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  #geom_point(aes(x=T, base_unf_accept_rate, color= "Unaugmented")) +
  #geom_line(aes(x=T, base_unf_accept_rate,color= "Unaugmented")) +
  geom_point(aes(x=rho, base_unf_accept_rate, color= "unfiltered")) +
  geom_line(aes(x=rho, base_unf_accept_rate,color= "unfiltered")) +
  geom_point(aes(x=rho, base_obs_accept_rate, color= "filtered:obs")) +
  geom_line(aes(x=rho, base_obs_accept_rate,color= "filtered:obs")) +
  geom_point(aes(x=rho, CP_accept_rate, color= "filtered:CP")) +
  geom_line(aes(x=rho, CP_accept_rate,color= "filtered:CP")) +
  facet_grid( lambda_obs_lab ~ temp )+
  ylab("Acceptance rate") + 
  xlab("rho") +
  #scale_x_log10()+
  theme_bw()

ggplot(summary_df %>% filter(delta ==0.5, T == 0.5,
                             lambda_obs < 0.1) %>%
         mutate(temp = paste0("T  = ", T),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  #geom_point(aes(x=T, base_unf_accept_rate, color= "Unaugmented")) +
  #geom_line(aes(x=T, base_unf_accept_rate,color= "Unaugmented")) +
  geom_point(aes(x=rho, base_unf_miscoverage, color= "unfiltered")) +
  geom_line(aes(x=rho, base_unf_miscoverage,color= "unfiltered")) +
  geom_point(aes(x=rho, base_obs_miscoverage, color= "filtered:obs")) +
  geom_line(aes(x=rho, base_obs_miscoverage,color= "filtered:obs")) +
  geom_point(aes(x=rho, CP_miscoverage, color= "filtered:CP")) +
  geom_line(aes(x=rho, CP_miscoverage,color= "filtered:CP")) +
  facet_grid( lambda_obs_lab ~ epsilon )+
  geom_abline(aes(intercept = 0.1, slope=0), colour="black")+
  ylab("Miscoverage rate") + 
  xlab("rho") +
  #scale_x_log10()+
  theme_bw()


ggplot(summary_df %>% filter(delta ==0.5, epsilon ==0.5,
                              rho ==0) %>%
         mutate(temp = paste0("T  = ", T),
                rho_lab = paste0("rho  = ", rho),
                n_lab = paste0("n_calib  = ", n_calib_docs),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  #geom_point(aes(x=T, base_unf_accept_rate, color= "Unaugmented")) +
  #geom_line(aes(x=T, base_unf_accept_rate,color= "Unaugmented")) +
  geom_point(aes(x=lambda_obs, base_unf_miscoverage, color= "unfiltered")) +
  geom_line(aes(x=lambda_obs, base_unf_miscoverage,color= "unfiltered")) +
  geom_point(aes(x=lambda_obs, base_obs_miscoverage, color= "filtered:obs")) +
  geom_line(aes(x=lambda_obs, base_obs_miscoverage,color= "filtered:obs")) +
  geom_point(aes(x=lambda_obs, CP_miscoverage, color= "filtered:CP")) +
  geom_line(aes(x=lambda_obs, CP_miscoverage,color= "filtered:CP")) +
  facet_grid( temp ~ n_lab , scales="free")+
  geom_abline(aes(intercept = 0.1, slope=0), colour="black")+
  ylab("Miscoverage rate") + 
  xlab("Lambda Obs") +
  scale_x_log10()+
  theme_bw()



colnames(summary_df)
ggplot(summary_df %>% filter(rho == 0, 
                             n_calib_docs ==200, delta ==0.5) %>%
         mutate(temp = paste0("T  = ", T),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=epsilon, reg_mse_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=epsilon, reg_mse_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=epsilon, reg_mse_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=epsilon, reg_mse_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=epsilon, reg_mse_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=epsilon, reg_mse_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=epsilon, reg_mse_aug_full_cp, color= "filtered:condCP")) +
  geom_line(aes(x=epsilon, reg_mse_aug_full_cp,color= "filtered:condCP")) +
  geom_point(aes(x=epsilon, reg_mse_aug_full_marginal, color= "filtered:margCP")) +
  geom_line(aes(x=epsilon, reg_mse_aug_full_marginal,color= "filtered:margCP")) +
  facet_grid( lambda_obs_lab ~ temp )+
  ylab("Reg MSE") + 
  scale_y_log10()+
  xlab("Epsilon") +
  theme_bw()


ggplot(summary_df %>% filter(rho == 10, delta ==0.5, lambda_obs<0.1) %>%
         mutate(temp = paste0("T  = ", T),
                epsilon_name = paste0("epsilon = ", epsilon),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=T, phi_l2_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=T, phi_l2_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=T, phi_l2_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=T, phi_l2_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=T, phi_l2_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=T, phi_l2_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=T, phi_l2_aug_full_cp, color= "filtered:CP")) +
  geom_line(aes(x=T, phi_l2_aug_full_cp,color= "filtered:CP")) +
  facet_grid( lambda_obs_lab ~ epsilon_name )+
  ylab("|| A_hat - A ||_2^2") + 
  xlab("Temperature") +
  scale_x_log10()+
  theme_bw()


ggplot(summary_df %>% filter(delta ==0.5, lambda_obs<0.1, epsilon==0.5) %>%
         mutate(temp = paste0("T  = ", T),
                epsilon_name = paste0("epsilon = ", epsilon),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  geom_point(aes(x=rho, phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=rho, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=rho, phi_l1_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=rho, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=rho, phi_l1_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=rho, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=rho, phi_l1_aug_full_cp, color= "filtered:CP")) +
  geom_line(aes(x=rho, phi_l1_aug_full_cp,color= "filtered:CP")) +
  facet_grid( lambda_obs_lab ~ temp )+
  ylab("|| A_hat - A ||_1") + 
  xlab("rho") +
  theme_bw()


ggplot(summary_df %>% filter(rho == 0, delta ==0.5) %>%
         mutate(temp = paste0("T  = ", T),
                lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))) + 
  #geom_point(aes(x=epsilon, base_unf_accept_rate, color= "Unaugmented")) +
  #geom_line(aes(x=epsilon, base_unf_accept_rate,color= "Unaugmented")) +
  geom_point(aes(x=epsilon, base_unf_accept_rate, color= "unfiltered")) +
  geom_line(aes(x=epsilon, base_unf_accept_rate,color= "unfiltered")) +
  geom_point(aes(x=epsilon, base_obs_accept_rate, color= "filtered:obs")) +
  geom_line(aes(x=epsilon, base_obs_accept_rate,color= "filtered:obs")) +
  geom_point(aes(x=epsilon, CP_accept_rate, color= "filtered:CP")) +
  geom_line(aes(x=epsilon, CP_accept_rate,color= "filtered:CP")) +
  facet_grid( lambda_obs_lab ~ temp )+
  ylab("Accept Rate") + 
  xlab("Epsilon") +
  theme_bw()


ggplot(summary_df %>% filter(delta == 0.1, 
                             epsilon ==0.5,
                             lambda_obs ==0.05)) + 
  #geom_point(aes(x=epsilon, base_unf_accept_rate, color= "Unaugmented")) +
  #geom_line(aes(x=epsilon, base_unf_accept_rate,color= "Unaugmented")) +
  geom_point(aes(x=T, base_unf_accept_rate, color= "unfiltered")) +
  geom_line(aes(x=T, base_unf_accept_rate,color= "unfiltered")) +
  geom_point(aes(x=T, base_obs_accept_rate, color= "filtered:obs")) +
  geom_line(aes(x=T, base_obs_accept_rate,color= "filtered:obs")) +
  geom_point(aes(x=T, cond_accept_rate, color= "filtered:CP")) +
  geom_line(aes(x=T, cond_accept_rate,color= "filtered:CP")) +
  facet_grid(rho ~ epsilon )+
  ylab("Accept Rate") + 
  xlab("Temperature") +
  theme_bw()



ggplot(summary_df %>% filter(delta == 0.1, 
                             rho ==1,
                             epsilon ==0.5)) + 
  #geom_point(aes(x=epsilon, base_unf_accept_rate, color= "Unaugmented")) +
  #geom_line(aes(x=epsilon, base_unf_accept_rate,color= "Unaugmented")) +
  geom_point(aes(x=lambda_obs, base_unf_accept_rate, color= "unfiltered")) +
  geom_line(aes(x=lambda_obs, base_unf_accept_rate,color= "unfiltered")) +
  geom_point(aes(x=lambda_obs, base_obs_accept_rate, color= "filtered:obs")) +
  geom_line(aes(x=lambda_obs, base_obs_accept_rate,color= "filtered:obs")) +
  geom_point(aes(x=lambda_obs, cond_accept_rate, color= "filtered:CP")) +
  geom_line(aes(x=lambda_obs, cond_accept_rate,color= "filtered:CP")) +
  facet_grid(. ~ T )+
  ylab("Accept Rate") + 
  xlab("lambda_obs") +
  theme_bw()




ggplot(summary_df %>% filter(T == 1.0, lambda_obs ==0.05)) + 
  geom_point(aes(x=epsilon, cond_accept_rate, color= "filtered:CP")) +
  geom_line(aes(x=epsilon, cond_accept_rate,color= "filtered:CP")) +
  facet_grid(rho ~ delta )+
  ylab("Accept Rate") + 
  ylab("Epsilon") +
  theme_bw()



