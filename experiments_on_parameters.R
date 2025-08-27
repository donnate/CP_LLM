library(dplyr)
library(readr)   # or data.table::fread if faster
library(purrr)   # for map_dfr
library(tidyverse)
# list all your results files (adjust pattern and path if needed)
files <- list.files(
  "/Users/clairedonnat/Documents/CP_LLM/results_new/",
  pattern = "experiment_on_alpha_.*\\.csv$",
  full.names = TRUE
)

files
# read and combine them all
results_df <- files %>%
  map_dfr(read_csv)   # or map_dfr(fread) if using data.table

unique(results_df$alpha)
# summarise like before
summary_df <- results_df %>%
  group_by(V, K, delta,alpha, epsilon, T, lambda_obs, rho, alpha_cp,
           n_train_docs, n_calib_docs) %>%
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
colnames(summary_df)


colnames(summary_df)

summary_df  =  summary_df%>%
  mutate(temp = paste0("T  = ", T),
         K_lab = paste0("K  = ", K),
         alpha_lab = paste0("alpha  = ", alpha),
         n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
         lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))
summary_df$K_lab = factor(summary_df$K_lab, levels=c( "K  = 3" , "K  = 10", "K  = 20"))

ggplot(summary_df %>% filter( epsilon ==0.5, 
                              rho==2,
                              lambda_obs == 0.05,
                              n_calib_docs == 50)) + 
  geom_point(aes(x=alpha, phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=alpha, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=alpha, phi_l1_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=alpha, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=alpha, phi_l1_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=alpha, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=alpha, phi_l1_aug_full_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=alpha , phi_l1_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=alpha, phi_l1_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=alpha , phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=alpha, phi_l1_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=alpha, phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid(K_lab  ~ temp , scales = "free" )+
  ylab("|| A_hat - A ||_1") + 
  scale_y_log10()+
  #scale_x_log10()+
  xlab("Alpha") +
  theme_bw()


ggplot(summary_df %>% filter( epsilon ==0.5, 
                              rho==2,
                              n_calib_docs == 50)) + 
  geom_point(aes(x=lambda_obs, phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=lambda_obs, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=lambda_obs, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=lambda_obs, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=lambda_obs , phi_l1_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=lambda_obs , phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=lambda_obs, phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid(K_lab  ~ alpha_lab , scales = "free" )+
  ylab("|| A_hat - A ||_1") + 
  scale_y_log10()+
  #scale_x_log10()+
  xlab("lambda_obs") +
  theme_bw()



colnames(summary_df)
ggplot(summary_df %>% filter( epsilon ==0.5, 
                              rho==2,
                              lambda_obs == 0.1,
                              n_calib_docs == 50)) + 
  geom_point(aes(x=alpha, n_added_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=alpha, n_added_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=alpha, n_added_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=alpha, n_added_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=alpha, n_added_obs, color= "filtered:obs")) +
  geom_line(aes(x=alpha, n_added_obs,color= "filtered:obs")) +
  geom_point(aes(x=alpha, n_added_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=alpha , n_added_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=alpha, n_added_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=alpha , n_added_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=alpha, n_added_marg, color= "filtered:CP_marg")) +
  geom_line(aes(x=alpha, n_added_marg,color= "filtered:CP_marg")) +
  facet_grid(K_lab  ~ temp , scales = "free" )+
  ylab("n added") + 
  #scale_y_log10()+
  #scale_x_log10()+
  xlab("alpha") +
  theme_bw()



colnames(summary_df)
ggplot(summary_df %>% filter( epsilon ==0.5, 
                              rho==2,
                              lambda_obs == 0.1,
                              n_calib_docs == 50)) + 
  geom_point(aes(x=alpha, 1, color= "Unaugmented")) +
  geom_line(aes(x=alpha, 1,color= "Unaugmented")) +
  geom_point(aes(x=alpha, cosine_mean_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=alpha, cosine_mean_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=alpha, cosine_mean_obs, color= "filtered:obs")) +
  geom_line(aes(x=alpha, cosine_mean_obs,color= "filtered:obs")) +
  geom_point(aes(x=alpha, cosine_mean_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=alpha , cosine_mean_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=alpha+ 0.01, cosine_mean_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=alpha+ 0.01 , cosine_mean_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=alpha, cosine_mean_marg, color= "filtered:CP_marg")) +
  geom_line(aes(x=alpha, cosine_mean_marg,color= "filtered:CP_marg")) +
  facet_grid(K_lab  ~ temp , scales = "free" )+
  ylab("Cosine similarity") + 
  #scale_y_log10()+
  #scale_x_log10()+
  xlab("alpha") +
  theme_bw()




files <- list.files(
  "/Users/clairedonnat/Documents/CP_LLM/results_new/",
  pattern = "experiment_on_K_.*\\.csv$",
  full.names = TRUE
)

files
# read and combine them all
results_df <- files %>%
  map_dfr(read_csv)   # or map_dfr(fread) if using data.table

unique(results_df$alpha)
# summarise like before
summary_df <- results_df %>%
  group_by(V, K, delta,alpha, epsilon, T, lambda_obs, rho, alpha_cp,
           n_train_docs, n_calib_docs) %>%
  summarise(across(where(is.numeric), median, na.rm = TRUE)) %>%
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
colnames(summary_df)


colnames(summary_df)

summary_df  =  summary_df%>%
  mutate(temp = paste0("T  = ", T),
         K_lab = paste0("K  = ", K),
         alpha_lab = paste0("alpha  = ", alpha),
         n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
         lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))
summary_df$K_lab = factor(summary_df$K_lab, levels=c( "K  = 3" , "K  = 5" ,
                                                      "K  = 10",  "K  = 15",  "K  = 20",
                                                      "K  = 30",  "K  = 40"))

ggplot(summary_df %>% filter( epsilon ==0.5, 
                              rho==2,
                              K>5,
                              lambda_obs == 0.05,
                              n_calib_docs == 50)) + 
  geom_point(aes(x=K, phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=K , phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=K+0.1, phi_l1_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=K+0.1, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=K+0.2, phi_l1_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=K+0.2, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=K+0.3, phi_l1_aug_full_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=K+0.3 , phi_l1_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=K+0.4, phi_l1_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=K+0.4 , phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=K+0.5, phi_l1_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=K+0.5, phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid( temp ~ alpha_lab , scales = "free" )+
  ylab("|| A_hat - A ||_1") + 
  scale_y_log10()+
  #scale_x_log10()+
  xlab("K") +
  theme_bw()


ggplot(summary_df %>% filter( epsilon ==0.5, 
                              rho==2,
                              alpha==1,
                              K %in% c(5, 15, 40),
                              n_calib_docs == 50)) + 
  geom_point(aes(x=lambda_obs, phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=lambda_obs, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=lambda_obs, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=lambda_obs, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=lambda_obs , phi_l1_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=lambda_obs , phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=lambda_obs, phi_l1_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=lambda_obs, phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid(K_lab  ~ alpha_lab , scales = "free" )+
  ylab("|| A_hat - A ||_1") + 
  #scale_y_log10()+
  #scale_x_log10()+
  xlab("lambda_obs") +
  theme_bw()


colnames(summary_df
         )
ggplot(summary_df %>% filter( epsilon ==0.5, 
                              rho==2,
                              alpha==0.1,
                              K %in% c(5, 15, 40),
                              n_calib_docs == 50)) + 
  geom_point(aes(x=1, phi_l1_unaug_full, color= "Unaugmented")) +
  #geom_line(aes(x=1, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=n_added_unfiltered+1, phi_l1_aug_full_unfiltered, shape=lambda_obs_lab, color= "unfiltered")) +
  #geom_line(aes(x=n_added_unfiltered, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=n_added_obs+1, phi_l1_aug_full_obs, shape=lambda_obs_lab, color= "filtered:obs")) +
  #geom_line(aes(x=n_added_obs, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=n_added_oracle+1, phi_l1_aug_full_oracle, shape=lambda_obs_lab, color= "filtered:oracle")) +
  #geom_line(aes(x=n_added_oracle , phi_l1_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=n_added_cp+1, phi_l1_aug_full_cp, shape=lambda_obs_lab, color= "filtered:CP_cond")) +
  #geom_line(aes(x=n_added_cp , phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=n_added_marg+1, phi_l1_aug_full_marginal, shape=lambda_obs_lab, color= "filtered:CP_marg")) +
  #geom_line(aes(x=n_added_marg, phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid(K_lab  ~ alpha_lab , scales = "free" )+
  ylab("|| A_hat - A ||_1") + 
  #scale_y_log10()+
  scale_x_log10()+
  xlab("Number Augmentations") +
  theme_bw()





files <- list.files(
  "/Users/clairedonnat/Documents/CP_LLM/results_new/",
  pattern = "new_synthetic_results_llm_cp_exp_exp_on_delta.*\\.csv$",
  full.names = TRUE
)

files
# read and combine them all
results_df <- files %>%
  map_dfr(read_csv)   # or map_dfr(fread) if using data.table

unique(results_df$alpha)
# summarise like before
summary_df <- results_df %>%
  group_by(V, K, delta,alpha, epsilon, T, lambda_obs, rho, alpha_cp,
           n_train_docs, n_calib_docs) %>%
  summarise(across(where(is.numeric), median, na.rm = TRUE)) %>%
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
colnames(summary_df)


colnames(summary_df)

summary_df  =  summary_df%>%
  mutate(temp = paste0("T  = ", T),
         K_lab = paste0("K  = ", K),
         alpha_lab = paste0("alpha  = ", alpha),
         n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
         lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))
unique(summary_df$K_lab)


ggplot(summary_df %>% filter( epsilon ==0.5, 
                              rho==2,
                              lambda_obs %in% c(0.005, 0.05, 0.1, 0.25),
                              #lambda_obs == 0.1,
                              n_calib_docs == 50)) + 
  geom_point(aes(x=delta-0.012, phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=delta-0.012 , phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=delta, phi_l1_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=delta, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=delta +0.012, phi_l1_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=delta +0.012, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=delta+0.02, phi_l1_aug_full_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=delta+0.02 , phi_l1_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=delta, phi_l1_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=delta, phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=delta+0.022, phi_l1_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=delta+0.022, phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid( alpha_lab +K_lab ~ lambda_obs_lab  , scales = "free" )+
  ylab("|| A_hat - A ||_1") + 
  scale_y_log10()+
  scale_x_log10()+
  xlab("Delta") +
  theme_bw()


find_max = summary_df %>%
  select(V, K, delta,alpha, epsilon, T, rho, alpha_cp,
         n_train_docs, n_calib_docs,
         lambda_obs, phi_l1_unaug_full,
         phi_l1_aug_full_unfiltered,
         phi_l1_aug_full_obs,
         phi_l1_aug_full_oracle,
         phi_l1_aug_full_cp, phi_l1_aug_full_marginal) %>%
  pivot_longer(cols=-c(V, K, delta,alpha, epsilon, T, rho, alpha_cp,
                       n_train_docs, n_calib_docs,
                       lambda_obs)) %>%
  group_by(V, K, delta,alpha, epsilon, T, rho, alpha_cp,
           n_train_docs, n_calib_docs,name) %>%
  slice_min(value)


find_max  =  find_max%>%
  mutate(temp = paste0("T  = ", T),
         K_lab = paste0("K  = ", K),
         alpha_lab = paste0("alpha  = ", alpha),
         n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
         lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))
ggplot(find_max %>% filter( epsilon ==0.5, 
                              rho==2,
                              K %in% c(5, 15, 40),
                              n_calib_docs == 50)) + 
  geom_point(aes(x=delta, value, color= name)) +
  geom_line(aes(x=delta, value,color= name)) +
  facet_grid(K_lab  ~ alpha_lab , scales = "free" )+
  ylab("|| A_hat - A ||_1") + 
  scale_y_log10()+
  scale_x_log10()+
  xlab("Delta") +
  theme_bw()


colnames(summary_df
)
ggplot(summary_df %>% filter( epsilon ==0.5, 
                              rho==2,
                              alpha==0.1,
                              K %in% c(5, 15, 40),
                              n_calib_docs == 50)) + 
  geom_point(aes(x=1, phi_l1_unaug_full, color= "Unaugmented")) +
  #geom_line(aes(x=1, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=n_added_unfiltered+1, phi_l1_aug_full_unfiltered, shape=lambda_obs_lab, color= "unfiltered")) +
  #geom_line(aes(x=n_added_unfiltered, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=n_added_obs+1, phi_l1_aug_full_obs, shape=lambda_obs_lab, color= "filtered:obs")) +
  #geom_line(aes(x=n_added_obs, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=n_added_oracle+1, phi_l1_aug_full_oracle, shape=lambda_obs_lab, color= "filtered:oracle")) +
  #geom_line(aes(x=n_added_oracle , phi_l1_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=n_added_cp+1, phi_l1_aug_full_cp, shape=lambda_obs_lab, color= "filtered:CP_cond")) +
  #geom_line(aes(x=n_added_cp , phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=n_added_marg+1, phi_l1_aug_full_marginal, shape=lambda_obs_lab, color= "filtered:CP_marg")) +
  #geom_line(aes(x=n_added_marg, phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid(K_lab  ~ alpha_lab , scales = "free" )+
  ylab("|| A_hat - A ||_1") + 
  #scale_y_log10()+
  scale_x_log10()+
  xlab("Number Augmentations") +
  theme_bw()







files <- list.files(
  "/Users/clairedonnat/Documents/CP_LLM/results_new/",
  pattern = "new_synthetic_results_llm_cp_exp_exp_on_n.*\\.csv$",
  full.names = TRUE
)

files
# read and combine them all
results_df <- files %>%
  map_dfr(read_csv)   # or map_dfr(fread) if using data.table

unique(results_df$alpha)
# summarise like before
summary_df <- results_df %>%
  group_by(V, K, delta,alpha, epsilon, T, lambda_obs, rho, alpha_cp,
           n_train_docs, n_calib_docs) %>%
  summarise(across(where(is.numeric), median, na.rm = TRUE)) %>%
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
colnames(summary_df)


colnames(summary_df)

summary_df  =  summary_df%>%
  mutate(temp = paste0("T  = ", T),
         K_lab = paste0("K  = ", K),
         alpha_lab = paste0("alpha  = ", alpha),
         n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
         lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))
unique(summary_df$K_lab)


ggplot(summary_df %>% filter( epsilon ==0.5, 
                              rho==2,
                              lambda_obs %in% c(0.005, 0.05, 0.1, 0.25)))+
                              #lambda_obs == 0.1)) + 
  geom_point(aes(x=n_calib_docs, y= phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=n_calib_docs , y= phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=n_calib_docs,  y=phi_l1_aug_full_unfiltered, color= "unfiltered")) +
  geom_line(aes(x=n_calib_docs,  y=phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=n_calib_docs,  y=phi_l1_aug_full_obs, color= "filtered:obs")) +
  geom_line(aes(x=n_calib_docs,  y=phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=n_calib_docs, y= phi_l1_aug_full_oracle, color= "filtered:oracle")) +
  geom_line(aes(x=n_calib_docs , y=phi_l1_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=n_calib_docs,  y=phi_l1_aug_full_cp, color= "filtered:CP_cond")) +
  geom_line(aes(x=n_calib_docs,  y=phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=n_calib_docs, y= phi_l1_aug_full_marginal, color= "filtered:CP_marg")) +
  geom_line(aes(x=n_calib_docs,  y=phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid( alpha_lab +K_lab ~ lambda_obs_lab  , scales = "free" )+
  ylab("|| A_hat - A ||_1") + 
  scale_y_log10()+
  scale_x_log10()+
  xlab("n_calib_docs") +
  theme_bw()


find_max = summary_df %>%
  select(V, K, delta,alpha, epsilon, T, rho, alpha_cp,
         n_train_docs, n_calib_docs,
         lambda_obs, phi_l1_unaug_full,
         phi_l1_aug_full_unfiltered,
         phi_l1_aug_full_obs,
         phi_l1_aug_full_oracle,
         phi_l1_aug_full_cp, phi_l1_aug_full_marginal) %>%
  pivot_longer(cols=-c(V, K, delta,alpha, epsilon, T, rho, alpha_cp,
                       n_train_docs, n_calib_docs,
                       lambda_obs)) %>%
  group_by(V, K, delta,alpha, epsilon, T, rho, alpha_cp,
           n_train_docs, n_calib_docs,name) %>%
  slice_min(value)

find_max  =  find_max%>%
  mutate(temp = paste0("T  = ", T),
         K_lab = paste0("K  = ", K),
         alpha_lab = paste0("alpha  = ", alpha),
         n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
         lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))
ggplot(find_max %>% filter( epsilon ==0.5, 
                            rho==2,
                            K %in% c(5, 15, 40))) + 
  geom_point(aes(x=n_calib_docs, value, color= name)) +
  geom_line(aes(x=n_calib_docs, value,color= name)) +
  facet_grid(K_lab  ~ alpha_lab , scales = "free" )+
  ylab("|| A_hat - A ||_1") + 
  scale_y_log10()+
  scale_x_log10()+
  xlab("Delta") +
  theme_bw()


colnames(summary_df
)
ggplot(summary_df %>% filter( epsilon ==0.5, 
                              rho==2,
                              alpha==0.1,
                              K %in% c(5, 15, 40))) + 
  geom_point(aes(x=1, phi_l1_unaug_full, color= "Unaugmented")) +
  #geom_line(aes(x=1, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=n_added_unfiltered+1, phi_l1_aug_full_unfiltered, size= n_calib_docs,shape=lambda_obs_lab, color= "unfiltered")) +
  #geom_line(aes(x=n_added_unfiltered, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=n_added_obs+1, phi_l1_aug_full_obs, shape=lambda_obs_lab, size= n_calib_docs, color= "filtered:obs")) +
  #geom_line(aes(x=n_added_obs, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=n_added_oracle+1, phi_l1_aug_full_oracle, shape=lambda_obs_lab, size= n_calib_docs, color= "filtered:oracle")) +
  #geom_line(aes(x=n_added_oracle , phi_l1_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=n_added_cp+1, phi_l1_aug_full_cp, shape=lambda_obs_lab, size= n_calib_docs, color= "filtered:CP_cond")) +
  #geom_line(aes(x=n_added_cp , phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=n_added_marg+1, phi_l1_aug_full_marginal, shape=lambda_obs_lab, size= n_calib_docs, color= "filtered:CP_marg")) +
  #geom_line(aes(x=n_added_marg, phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid(K_lab  ~ alpha_lab , scales = "free" )+
  ylab("|| A_hat - A ||_1") + 
  #scale_y_log10()+
  scale_x_log10()+
  xlab("Number Augmentations") +
  theme_bw()




files <- list.files(
  "/Users/clairedonnat/Documents/CP_LLM/results_new/",
  pattern = "new_synthetic_results_llm_cp_exp_temperature.*\\.csv$",
  full.names = TRUE
)

files
# read and combine them all
results_df <- files %>%
  map_dfr(read_csv)   # or map_dfr(fread) if using data.table

unique(results_df$alpha)
# summarise like before
summary_df <- results_df %>%
  group_by(V, K, delta,alpha, epsilon, T, lambda_obs, rho, alpha_cp,
           n_train_docs, n_calib_docs) %>%
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
colnames(summary_df)


colnames(summary_df)

summary_df  =  summary_df%>%
  mutate(temp = paste0("T  = ", T),
         K_lab = paste0("K  = ", K),
         alpha_lab = paste0("alpha  = ", alpha),
         n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
         lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))
unique(summary_df$K_lab)


ggplot(summary_df %>% filter( epsilon ==0.5, 
                              rho==2,
                              lambda_obs %in% c(0.005, 0.05, 0.1, 0.25)))+
  #lambda_obs == 0.1)) + 
  geom_point(aes(x=T, y= phi_l1_unaug_full, color= "Unaugmented")) +
  geom_line(aes(x=T , y= phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=T,  y=phi_l1_aug_full_unfiltered, size=n_added_unfiltered+1,  color= "unfiltered")) +
  geom_line(aes(x=T,  y=phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=T,  y=phi_l1_aug_full_obs, size=n_added_obs+1, color= "filtered:obs")) +
  geom_line(aes(x=T,  y=phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=T, y= phi_l1_aug_full_oracle, size=n_added_oracle+1, color= "filtered:oracle")) +
  geom_line(aes(x=T , y=phi_l1_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=T,  y=phi_l1_aug_full_cp,  size=n_added_cp+1,color= "filtered:CP_cond")) +
  geom_line(aes(x=T,  y=phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=T, y= phi_l1_aug_full_marginal,  size=n_added_marg+1, color= "filtered:CP_marg")) +
  geom_line(aes(x=T,  y=phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid( alpha_lab +K_lab ~ lambda_obs_lab  , scales = "free" )+
  ylab("|| A_hat - A ||_1") + 
  scale_y_log10()+
  scale_x_log10()+
  xlab("T") +
  theme_bw()


find_max = summary_df %>%
  select(V, K, delta,alpha, epsilon, T, rho, alpha_cp,
         n_train_docs, n_calib_docs,
         lambda_obs, phi_l1_unaug_full,
         phi_l1_aug_full_unfiltered,
         phi_l1_aug_full_obs,
         phi_l1_aug_full_oracle,
         phi_l1_aug_full_cp, phi_l1_aug_full_marginal) %>%
  pivot_longer(cols=-c(V, K, delta,alpha, epsilon, T, rho, alpha_cp,
                       n_train_docs, n_calib_docs,
                       lambda_obs)) %>%
  group_by(V, K, delta,alpha, epsilon, T, rho, alpha_cp,
           n_train_docs, n_calib_docs,name) %>%
  slice_min(value)

find_max  =  find_max%>%
  mutate(temp = paste0("T  = ", T),
         K_lab = paste0("K  = ", K),
         alpha_lab = paste0("alpha  = ", alpha),
         n_calib_docs_lab = paste0("n_calib_docs = ", n_calib_docs),
         lambda_obs_lab = paste0("lambda_obs = ", lambda_obs))
ggplot(find_max %>% filter( epsilon ==0.5, 
                            rho==2,
                            K %in% c(5, 15, 40))) + 
  geom_point(aes(x=T, value, color= name)) +
  geom_line(aes(x=T, value,color= name)) +
  facet_grid(K_lab  ~ alpha_lab , scales = "free" )+
  ylab("|| A_hat - A ||_1") + 
  scale_y_log10()+
  scale_x_log10()+
  xlab("T") +
  theme_bw()


colnames(summary_df
)
ggplot(summary_df %>% filter( epsilon ==0.5, 
                              rho==2,
                              #alpha==0.1,
                              K %in% c(5, 15, 40))) + 
  geom_point(aes(x=1, phi_l1_unaug_full, color= "Unaugmented")) +
  #geom_line(aes(x=1, phi_l1_unaug_full,color= "Unaugmented")) +
  geom_point(aes(x=n_added_unfiltered+1, phi_l1_aug_full_unfiltered, size= T,shape=lambda_obs_lab, color= "unfiltered")) +
  #geom_line(aes(x=n_added_unfiltered, phi_l1_aug_full_unfiltered,color= "unfiltered")) +
  geom_point(aes(x=n_added_obs+1, phi_l1_aug_full_obs, shape=lambda_obs_lab, size= T, color= "filtered:obs")) +
  #geom_line(aes(x=n_added_obs, phi_l1_aug_full_obs,color= "filtered:obs")) +
  geom_point(aes(x=n_added_oracle+1, phi_l1_aug_full_oracle, shape=lambda_obs_lab, size= T, color= "filtered:oracle")) +
  #geom_line(aes(x=n_added_oracle , phi_l1_aug_full_oracle,color= "filtered:oracle")) +
  geom_point(aes(x=n_added_cp+1, phi_l1_aug_full_cp, shape=lambda_obs_lab, size= T, color= "filtered:CP_cond")) +
  #geom_line(aes(x=n_added_cp , phi_l1_aug_full_cp,color= "filtered:CP_cond")) +
  geom_point(aes(x=n_added_marg+1, phi_l1_aug_full_marginal, shape=lambda_obs_lab, size= T, color= "filtered:CP_marg")) +
  #geom_line(aes(x=n_added_marg, phi_l1_aug_full_marginal,color= "filtered:CP_marg")) +
  facet_grid(K_lab  ~ alpha_lab , scales = "free" )+
  ylab("|| A_hat - A ||_1") + 
  #scale_y_log10()+
  scale_x_log10()+
  xlab("Number Augmentations") +
  theme_bw()




