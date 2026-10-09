# v2 run status — 2026-10-09 06:25 UTC

## Pipeline
```
parsed ok: 34302/34302; notices: 20032
median body words: 5039; >=3 sections: 0.939
docs with leak sentences removed: 0.566
PHASE_A_DONE Wed Oct  7 06:44:42 PM UTC 2026
2026-10-07T18:45:56Z A6 start
2026-10-07T18:47:33Z A6 done; D1 fulltext audit
2026-10-07T18:49:08Z FT_READY
FT_READY: True
```

## GPU queue (GPU 1)
```
2026-10-08T17:45:41Z START 153_abs_ditto_s42
2026-10-08T19:08:38Z DONE 153_abs_ditto_s42
2026-10-08T19:08:38Z START 154_abs_ditto_s43
2026-10-08T20:31:27Z DONE 154_abs_ditto_s43
2026-10-08T20:31:27Z START 155_abs_ditto_s44
2026-10-08T21:54:07Z DONE 155_abs_ditto_s44
2026-10-08T21:54:07Z START 160_ft8k_s42_central
2026-10-08T23:08:11Z DONE 160_ft8k_s42_central
2026-10-08T23:08:11Z START 161_ft8k_s42_fedavg
2026-10-09T02:50:40Z DONE 161_ft8k_s42_fedavg
2026-10-09T02:50:40Z START 162_ft8k_s42_local
2026-10-09T04:15:19Z DONE 162_ft8k_s42_local
pending: 
failed: 020_emb_abs_qwen8b
1, 17790 MiB, 99 %
```

GPU jobs started: 54, done: 53, failed: 1

## B2 main benchmark (pooled test, AUPRC, 10 seeds)
```
                                    mean     std
setting partition model                         
pub     field     central_logreg  0.3733  0.0036
                  central_mlp     0.5077  0.0033
                  central_rf      0.4863  0.0030
                  central_xgb     0.5103  0.0035
                  fl_fedavg       0.4885  0.0034
                  fl_fedbal       0.4713  0.0032
                  fl_fedprox      0.4870  0.0037
                  fl_scaffold     0.4710  0.0050
                  fl_xgb_cyclic   0.4543  0.0039
        publisher central_logreg  0.3739  0.0023
                  central_mlp     0.5107  0.0056
                  central_rf      0.4858  0.0038
                  central_xgb     0.5124  0.0048
                  fl_fedavg       0.4809  0.0046
                  fl_fedbal       0.4745  0.0040
                  fl_fedprox      0.4772  0.0050
                  fl_scaffold     0.4558  0.0103
                  fl_xgb_cyclic   0.4100  0.0051
pub2    field     central_logreg  0.1971  0.0045
                  central_mlp     0.3447  0.0080
                  central_rf      0.3386  0.0073
                  central_xgb     0.3602  0.0052
                  fl_fedavg       0.3034  0.0130
                  fl_fedbal       0.3016  0.0078
                  fl_fedprox      0.3056  0.0108
                  fl_scaffold     0.2935  0.0071
                  fl_xgb_cyclic   0.1992  0.0105
        publisher central_logreg  0.1992  0.0065
                  central_mlp     0.3520  0.0136
                  central_rf      0.3447  0.0109
                  central_xgb     0.3683  0.0131
                  fl_fedavg       0.3230  0.0107
                  fl_fedbal       0.3080  0.0113
                  fl_fedprox      0.3270  0.0084
                  fl_scaffold     0.3180  0.0081
                  fl_xgb_cyclic   0.2490  0.0067
v1leak  field     central_logreg  0.3741  0.0036
                  central_mlp     0.5277  0.0034
                  central_rf      0.5434  0.0034
                  central_xgb     0.5435  0.0031
                  fl_fedavg       0.5040  0.0028
                  fl_fedbal       0.4826  0.0026
                  fl_fedprox      0.5002  0.0032
                  fl_scaffold     0.4824  0.0026
                  fl_xgb_cyclic   0.4873  0.0036
        publisher central_logreg  0.3747  0.0022
                  central_mlp     0.5316  0.0063
                  central_rf      0.5425  0.0021
                  central_xgb     0.5437  0.0054
                  fl_fedavg       0.4936  0.0039
                  fl_fedbal       0.4815  0.0024
                  fl_fedprox      0.4891  0.0040
                  fl_scaffold     0.4651  0.0062
                  fl_xgb_cyclic   0.4433  0.0077
```

## B2 own-silo view (mean over silos)
```
model              central_logreg  central_mlp  central_rf  central_xgb  fl_fedavg  fl_fedavg_ft  fl_fedbal  fl_fedprox  fl_scaffold  fl_xgb_cyclic  local_mlp  local_xgb
setting partition                                                                                                                                                        
pub     field              0.3706       0.4966      0.4760       0.5022     0.4878        0.5239     0.4691      0.4880       0.4765         0.4583     0.5138     0.5154
        publisher          0.3396       0.4907      0.4868       0.4890     0.4699        0.5562     0.4596      0.4681       0.4766         0.4315     0.5759     0.5752
pub2    field              0.1753       0.2714      0.2728       0.2766     0.2565        0.2960     0.2331      0.2638       0.2578         0.2013     0.2714     0.2889
        publisher          0.2097       0.3744      0.3769       0.3833     0.3464        0.4142     0.3236      0.3476       0.3621         0.2892     0.4277     0.4509
v1leak  field              0.3724       0.5244      0.5398       0.5429     0.5095        0.5463     0.4849      0.5064       0.4932         0.4974     0.5281     0.5481
        publisher          0.3408       0.5118      0.5253       0.5134     0.4827        0.5700     0.4681      0.4821       0.4865         0.4466     0.5880     0.5995
```

## B3 ablations (pooled AUPRC)
```
                                 mean     std  count
exp              model                              
pub_all          central_mlp   0.5107  0.0056     10
                 central_xgb   0.5124  0.0048     10
                 fl_fedavg     0.4809  0.0046     10
pub_has_abstract central_mlp   0.5188  0.0070     10
                 central_xgb   0.5186  0.0056     10
                 fl_fedavg     0.4997  0.0066     10
pub_meta_only    central_mlp   0.4683  0.0040     10
                 central_xgb   0.4769  0.0036     10
                 fl_fedavg     0.4530  0.0029     10
pub_no_year      central_mlp   0.4716  0.0049     10
                 central_xgb   0.4696  0.0049     10
                 fl_fedavg     0.4518  0.0064     10
pub_silo_onehot  central_mlp   0.5777  0.0060     10
                 central_xgb   0.5760  0.0063     10
pub_text_only    central_mlp   0.3735  0.0035     10
                 central_xgb   0.3666  0.0041     10
                 fl_fedavg     0.3631  0.0035     10
v1exact          central_mlp   0.6370  0.0044     10
                 central_xgb   0.6508  0.0042     10
                 fl_fedavg     0.5800  0.0084     10
                 local_logreg  0.3403  0.0026     10
```

## B4 scale & DP (pooled AUPRC)
```
                                          mean     std  count
exp          k    q   sigma epsilon                          
dp_journal   NaN  0.1 0.0   inf         0.3739  0.0067     10
                      0.5   50.301726   0.3388  0.0146     10
                      1.0   11.015671   0.3202  0.0110     10
                      2.0   3.679746    0.2938  0.0131     10
                      4.0   1.556808    0.2875  0.0160     10
dp_publisher NaN  1.0 0.0   inf         0.4451  0.0094     10
                      0.3   365.168742  0.4442  0.0104     10
                      0.5   150.684966  0.4278  0.0137     10
                      1.0   49.405968   0.3747  0.0224     10
                      2.0   18.021461   0.2900  0.0096     10
journal      NaN  0.1 NaN   NaN         0.4188  0.0161     10
                  1.0 NaN   NaN         0.4010  0.0111     10
ksweep       5.0  NaN NaN   NaN         0.4716  0.0064     10
             10.0 NaN NaN   NaN         0.4809  0.0046     10
             15.0 NaN NaN   NaN         0.4783  0.0050     10
             20.0 NaN NaN   NaN         0.4794  0.0062     10
```

## B5 prospective
```
             auprc  roc_auc
model                      
central_xgb  0.071    0.713
fl_fedavg    0.056    0.617
```

## B6 cross-corpus
```
                                                                              accuracy  accuracy_at_0.5  auprc  roc_auc
exp                 arena      features          train           model                                                 
T1_to_ours          NaN        NaN               curated_180     NaN               NaN              NaN  0.299    0.504
                                                 ours_random_180 NaN               NaN              NaN  0.311    0.529
                                                 ours_random_464 NaN               NaN              NaN  0.316    0.536
T23_ours_to_curated NaN        all               NaN             central_xgb       NaN            0.573  0.518    0.573
                                                                 fl_fedavg         NaN            0.524  0.484    0.532
                               meta_only         NaN             central_xgb       NaN            0.513  0.485    0.529
                                                                 fl_fedavg         NaN            0.519  0.494    0.525
                               text_only         NaN             central_xgb       NaN            0.536  0.490    0.532
                                                                 fl_fedavg         NaN            0.518  0.472    0.517
T4_cv5x3            audited172 his_abstract_only NaN             NaN             0.622              NaN  0.607      NaN
                               his_all_10        NaN             NaN             0.917              NaN  0.975      NaN
T4_his_protocol     all464     his_abstract_only NaN             NaN             0.624              NaN  0.653    0.655
                               his_all_10        NaN             NaN             0.978              NaN  0.995    0.995
                    clean180   his_abstract_only NaN             NaN             0.639              NaN  0.704    0.722
                               his_all_10        NaN             NaN             0.889              NaN  0.946    0.941
```

## C2 embedding heads (AUPRC)
```
                                                                              mean     std  count
source    emb                          with_tab view        model                                
abstracts abstracts_modernbert         False    pooled      central_logreg  0.6507  0.0045     10
                                                            central_mlp     0.6864  0.0037     10
                                                            fl_fedavg       0.6640  0.0083     10
                                       True     pooled      central_logreg  0.6595  0.0042     10
                                                            central_mlp     0.7040  0.0049     10
                                                            fl_fedavg       0.6748  0.0055     10
          abstracts_qwen3emb8b         False    pooled      central_logreg  0.6937  0.0053     10
                                                            central_mlp     0.6877  0.0057     10
                                                            fl_fedavg       0.6712  0.0077     10
                                       True     pooled      central_logreg  0.7009  0.0052     10
                                                            central_mlp     0.6968  0.0070     10
                                                            fl_fedavg       0.6724  0.0064     10
fulltext  fulltext_modernbert_sections False    pooled      central_logreg  0.5830  0.0123     10
                                                            central_mlp     0.6171  0.0127     10
                                                            fl_fedavg       0.5914  0.0157     10
                                                prospective central_logreg  0.7781  0.0497     10
                                                            central_mlp     0.8128  0.0541     10
                                                            fl_fedavg       0.7282  0.0435     10
                                       True     pooled      central_logreg  0.6016  0.0095     10
                                                            central_mlp     0.6342  0.0152     10
                                                            fl_fedavg       0.6055  0.0207     10
                                                prospective central_logreg  0.8240  0.0375     10
                                                            central_mlp     0.8373  0.0454     10
                                                            fl_fedavg       0.8196  0.0523     10
          fulltext_qwen3emb4b_sections False    pooled      central_logreg  0.5356  0.0133     10
                                                            central_mlp     0.5916  0.0143     10
                                                            fl_fedavg       0.5616  0.0167     10
                                                prospective central_logreg  0.7256  0.0438     10
                                                            central_mlp     0.8406  0.0450     10
                                                            fl_fedavg       0.8097  0.0515     10
                                       True     pooled      central_logreg  0.5523  0.0151     10
                                                            central_mlp     0.5985  0.0140     10
                                                            fl_fedavg       0.5729  0.0196     10
                                                prospective central_logreg  0.7505  0.0413     10
                                                            central_mlp     0.8076  0.0631     10
                                                            fl_fedavg       0.8412  0.0474     10
```

## c3_lora_abstracts (LoRA fine-tuning; local_lora pooled = mean over silo models)
```
                                 auprc       roc_auc      
                                  mean count    mean count
model              view                                   
central_lora       pooled       0.7517     3  0.8563     3
                   prospective  0.1872     3  0.8827     3
dpsgd_central_lora pooled       0.6229     2  0.7761     2
                   prospective  0.1496     2  0.8292     2
dpsgd_fedavg_lora  pooled       0.5423     1  0.7220     1
                   prospective  0.1258     1  0.7997     1
fl_ditto_lora      pooled       0.6757    30  0.8093    30
fl_fedavg_lora     pooled       0.7195     6  0.8376     6
                   prospective  0.1939     6  0.8759     6
fl_fedavg_lora_ft  pooled       0.6789    30  0.8154    30
fl_fedper_lora     pooled       0.6677    30  0.8031    30
local_lora         pooled       0.5477    30  0.7318    30
                   prospective  0.1122    30  0.8048    30
```

own-silo view:
```
                     auprc
model                     
central_lora        0.7426
dpsgd_central_lora  0.6441
dpsgd_fedavg_lora   0.5662
fl_ditto_lora       0.7677
fl_fedavg_lora      0.7172
fl_fedavg_lora_ft   0.7736
fl_fedper_lora      0.7614
local_lora          0.7612
```

## c3_lora (LoRA fine-tuning; local_lora pooled = mean over silo models)
```
                                auprc       roc_auc      
                                 mean count    mean count
model             view                                   
central_lora      pooled       0.6040     3  0.7426     3
                  prospective  0.8121     3  0.7730     3
fl_dp_lora        pooled       0.3541     2  0.4882     2
                  prospective  0.6745     2  0.5976     2
fl_fedavg_lora    pooled       0.5860     3  0.7282     3
                  prospective  0.7557     3  0.7571     3
fl_fedavg_lora_ft pooled       0.5826    24  0.7267    24
local_lora        pooled       0.4894    24  0.6362    24
                  prospective  0.6836    24  0.6785    24
```

own-silo view:
```
                    auprc
model                    
central_lora       0.6032
fl_dp_lora         0.3617
fl_fedavg_lora     0.5738
fl_fedavg_lora_ft  0.5945
local_lora         0.5790
```

## d1_leak_audit_abstracts.txt
```
[title] keyword-only classifier: {'auprc': 0.2901154857221515, 'roc_auc': 0.5005654853381475, 'recall_at_5fpr': 0.00031210986267166043}
[title] TF-IDF logistic regression (5-fold): {'auprc': 0.5990331303897333, 'roc_auc': 0.7756319298920777, 'recall_at_5fpr': 0.282970150947679}
[title] notice-like tokens among top-300 positive n-grams: none
[abstract] keyword-only classifier: {'auprc': 0.2937033119557769, 'roc_auc': 0.5033084388053386, 'recall_at_5fpr': 0.026841448189762796}
[abstract] TF-IDF logistic regression (5-fold): {'auprc': 0.5960260180410544, 'roc_auc': 0.7650697365186184, 'recall_at_5fpr': 0.3287084326410169}
[abstract] notice-like tokens among top-300 positive n-grams: none
[title+abstract] keyword-only classifier: {'auprc': 0.2930584082802138, 'roc_auc': 0.503372905091723, 'recall_at_5fpr': 0.03160821700147543}
[title+abstract] TF-IDF logistic regression (5-fold): {'auprc': 0.662809860093528, 'roc_auc': 0.8145077925438914, 'recall_at_5fpr': 0.3855691748950176}
[title+abstract] notice-like tokens among top-300 positive n-grams: none
```

## d1_leak_audit_fulltext.txt
```
[title+abstract] keyword-only classifier: {'auprc': 0.36181278599419364, 'roc_auc': 0.5046895486971996, 'recall_at_5fpr': 0.04433497536945813}
[title+abstract] TF-IDF logistic regression (5-fold): {'auprc': 0.5911033383268605, 'roc_auc': 0.7322520594262073, 'recall_at_5fpr': 0.20689655172413793}
[title+abstract] notice-like tokens among top-300 positive n-grams: none
[methods] keyword-only classifier: {'auprc': 0.37739058739750403, 'roc_auc': 0.5250140264797339, 'recall_at_5fpr': 0.06916256157635468}
[methods] TF-IDF logistic regression (5-fold): {'auprc': 0.628108216320951, 'roc_auc': 0.7622796560633288, 'recall_at_5fpr': 0.2465024630541872}
[methods] notice-like tokens among top-300 positive n-grams: none
[body] keyword-only classifier: {'auprc': 0.3768226897399421, 'roc_auc': 0.5278193550209581, 'recall_at_5fpr': 0.04315270935960591}
[body] TF-IDF logistic regression (5-fold): {'auprc': 0.6403594708493894, 'roc_auc': 0.7734763505511723, 'recall_at_5fpr': 0.26108374384236455}
[body] notice-like tokens among top-300 positive n-grams: none
```
