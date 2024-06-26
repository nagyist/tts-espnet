# SPRING-INX Recipe
ESPnet Recipes to get started with SPRING-INX data.

# Data Statistics

We are releasing the first set of valuable data amounting to 2000 hours (both Audio and corresponding manually transcribed data) which was collected, cleaned and prepared for ASR system building in 10 Indian languages such as Assamese, Bengali Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi and Tamil in the public domain.

The prepared language-wise dataset was then split to train, valid and test tests. The number hours of training and validation data per language is presented in following table.

| Language  | Train | Valid | Test | Total (Approx.) |
|-----------|-------|-------|------|-----------------|
| Assamese  | 50.6  | 5.1   | 5.0  | 61              |
| Bengali   | 374.7 | 40.0  | 5.0  | 420             |
| Gujarati  | 175.5 | 19.6  | 5.0  | 200             |
| Hindi     | 316.4 | 29.7  | 5.0  | 351             |
| Kannada   | 82.5  | 9.7   | 4.8  | 97              |
| Malayalam | 214.7 | 24.7  | 5.0  | 245             |
| Marathi   | 130.4 | 14.4  | 5.2  | 150             |
| Odia      | 82.5  | 9.3   | 4.7  | 96              |
| Punjabi   | 140.0 | 15.1  | 5.1  | 159             |
| Tamil     | 200.7 | 20.0  | 5.1  | 226             |

For more information you can refer the arVix paper : https://arxiv.org/abs/2310.14654v2

# Usage

 You can select the language you want by specifying the "lang" parameter in the run.sh file. By default, it is set to "assamese".

# Results
# asr_assamese
## Environments
- date: `Mon Jun 10 11:38:24 IST 2024`
- python version: `3.8.18 (default, Sep 11 2023, 13:40:15)  [GCC 11.2.0]`
- espnet version: `espnet 202308`
- pytorch version: `pytorch 2.0.1`
- Git hash: `a10a6ae84f75a0c81ffa2c5435746b109fa5bf7f`
  - Commit date: `Mon Jun 10 11:36:30 2024 +0530`
- ASR config: [conf/tuning/train_asr_transformer.yaml](conf/tuning/train_asr_transformer.yaml)
- Decode config: [conf/decode_asr.yaml](conf/decode_asr.yaml)
- nbpe: 200

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|3845|39602|55.9|38.1|6.0|6.1|50.3|92.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|3845|203618|82.9|10.1|6.9|7.1|24.2|92.0|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|3845|169247|81.1|11.0|7.9|6.1|25.0|87.3|

<!-- Generated by ./scripts/utils/show_asr_result.sh -->
# asr_bengali
## Environments
- date: `Mon Jun 10 11:59:00 IST 2024`
- python version: `3.8.18 (default, Sep 11 2023, 13:40:15)  [GCC 11.2.0]`
- espnet version: `espnet 202308`
- pytorch version: `pytorch 1.13.1`
- Git hash: `13f42bc21d298923ac41cac677382f228697da24`
  - Commit date: `Mon Jun 10 11:56:01 2024 +0530`
- ASR config: [conf/tuning/train_asr_transformer.yaml](conf/tuning/train_asr_transformer.yaml)
- Decode config: [conf/decode_asr.yaml](conf/decode_asr.yaml)
- nbpe: 1300

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|2196|38201|69.1|26.4|4.5|5.6|36.5|93.5|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|2196|202714|90.1|5.5|4.4|6.2|16.0|93.5|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|2196|68202|77.7|13.6|8.8|5.5|27.8|92.9|

<!-- Generated by ./scripts/utils/show_asr_result.sh -->
# asr_gujarati
## Environments
- date: `Mon Jun 10 12:02:20 IST 2024`
- python version: `3.8.18 (default, Sep 11 2023, 13:40:15)  [GCC 11.2.0]`
- espnet version: `espnet 202308`
- pytorch version: `pytorch 1.13.1`
- Git hash: `04084cf8eb396fde57a3a886433cdeef92796115`
  - Commit date: `Mon Jun 10 12:02:01 2024 +0530`
- ASR config: [conf/tuning/train_asr_transformer.yaml](conf/tuning/train_asr_transformer.yaml)
- Decode config: [conf/decode_asr.yaml](conf/decode_asr.yaml)
- nbpe: 600

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|1994|37894|67.3|26.9|5.7|9.1|41.8|93.9|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|1994|186082|87.5|6.5|6.0|8.7|21.2|93.9|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|1994|87758|76.8|15.4|7.8|8.8|32.0|94.0|

<!-- Generated by ./scripts/utils/show_asr_result.sh -->
# asr_hindi
## Environments
- date: `Mon Jun 10 12:19:16 IST 2024`
- python version: `3.8.18 (default, Sep 11 2023, 13:40:15)  [GCC 11.2.0]`
- espnet version: `espnet 202308`
- pytorch version: `pytorch 1.13.1`
- Git hash: `877cf85df76424805950c6290206c1e6ac926ca5`
  - Commit date: `Mon Jun 10 12:19:05 2024 +0530`
- ASR config: [conf/tuning/train_asr_transformer.yaml](conf/tuning/train_asr_transformer.yaml)
- Decode config: [conf/decode_asr.yaml](conf/decode_asr.yaml)
- nbpe: 1100

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|1286|23262|76.0|19.8|4.2|5.8|29.8|90.6|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|1286|103103|90.0|5.2|4.8|6.1|16.0|90.6|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|1286|35335|77.7|14.0|8.3|5.6|27.9|90.3|

<!-- Generated by ./scripts/utils/show_asr_result.sh -->
# asr_kannada
## Environments
- date: `Mon Jun 10 13:40:37 IST 2024`
- python version: `3.8.18 (default, Sep 11 2023, 13:40:15)  [GCC 11.2.0]`
- espnet version: `espnet 202308`
- pytorch version: `pytorch 1.13.1`
- Git hash: `b4ceaeeb7ff82f9175c7515c20a03269f4ca9065`
  - Commit date: `Mon Jun 10 13:40:20 2024 +0530`
- ASR config: [conf/tuning/train_asr_transformer.yaml](conf/tuning/train_asr_transformer.yaml)
- Decode config: [conf/decode_asr.yaml](conf/decode_asr.yaml)
- nbpe: 300

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|4314|33295|51.2|41.5|7.2|7.2|55.9|87.9|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|4314|212401|83.7|8.2|8.1|7.7|23.9|87.9|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|4314|141260|76.8|13.7|9.6|7.5|30.8|87.9|

<!-- Generated by ./scripts/utils/show_asr_result.sh -->
# asr_malayalam
## Environments
- date: `Mon Jun 10 12:22:05 IST 2024`
- python version: `3.8.18 (default, Sep 11 2023, 13:40:15)  [GCC 11.2.0]`
- espnet version: `espnet 202308`
- pytorch version: `pytorch 1.13.1`
- Git hash: `acc616892db822275d8f5e98a4891daa48bdbbde`
  - Commit date: `Mon Jun 10 12:21:44 2024 +0530`
- ASR config: [conf/tuning/train_asr_transformer.yaml](conf/tuning/train_asr_transformer.yaml)
- Decode config: [conf/decode_asr.yaml](conf/decode_asr.yaml)
- nbpe: 750

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|2389|28199|60.4|33.7|5.9|7.2|46.8|92.1|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|2389|239593|89.5|5.4|5.1|5.1|15.5|92.1|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|2389|96006|77.6|14.7|7.7|4.9|27.4|91.5|

<!-- Generated by ./scripts/utils/show_asr_result.sh -->
# asr_marathi
## Environments
- date: `Mon Jun 10 13:37:38 IST 2024`
- python version: `3.8.18 (default, Sep 11 2023, 13:40:15)  [GCC 11.2.0]`
- espnet version: `espnet 202308`
- pytorch version: `pytorch 1.13.1`
- Git hash: `0f7c092c11b88999f3c637dd36f6f30b704d3370`
  - Commit date: `Mon Jun 10 12:30:43 2024 +0530`
- ASR config: [conf/tuning/train_asr_transformer.yaml](conf/tuning/train_asr_transformer.yaml)
- Decode config: [conf/decode_asr.yaml](conf/decode_asr.yaml)
- nbpe: 450

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|3267|37047|58.5|30.4|11.1|7.2|48.7|84.5|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|3267|222185|80.8|7.5|11.6|7.2|26.4|84.5|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|3267|125418|73.9|13.6|12.5|6.8|32.9|84.4|

<!-- Generated by ./scripts/utils/show_asr_result.sh -->
# asr_odia
## Environments
- date: `Mon Jun 10 13:43:36 IST 2024`
- python version: `3.8.18 (default, Sep 11 2023, 13:40:15)  [GCC 11.2.0]`
- espnet version: `espnet 202308`
- pytorch version: `pytorch 1.13.1`
- Git hash: `f0ff227f29328a690aaafe70a90b861505e465b9`
  - Commit date: `Mon Jun 10 13:43:10 2024 +0530`
- ASR config: [conf/tuning/train_asr_transformer.yaml](conf/tuning/train_asr_transformer.yaml)
- Decode config: [conf/decode_asr.yaml](conf/decode_asr.yaml)
- nbpe: 300

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|3511|38105|60.2|33.4|6.4|9.0|48.8|91.7|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|3511|196044|82.7|9.5|7.8|7.9|25.3|91.7|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|3511|195492|82.5|9.7|7.8|7.8|25.4|91.7|

<!-- Generated by ./scripts/utils/show_asr_result.sh -->
# asr_punjabi
## Environments
- date: `Mon Jun 10 13:46:04 IST 2024`
- python version: `3.8.18 (default, Sep 11 2023, 13:40:15)  [GCC 11.2.0]`
- espnet version: `espnet 202308`
- pytorch version: `pytorch 1.13.1`
- Git hash: `7632431665607ab644d98004921f7d8acbbe78d6`
  - Commit date: `Mon Jun 10 13:45:47 2024 +0530`
- ASR config: [conf/tuning/train_asr_transformer.yaml](conf/tuning/train_asr_transformer.yaml)
- Decode config: [conf/decode_asr.yaml](conf/decode_asr.yaml)
- nbpe: 500

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|2797|45195|73.5|22.5|4.0|5.3|31.8|84.0|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|2797|209038|89.4|5.8|4.7|6.0|16.6|84.0|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|2797|119527|83.0|10.9|6.2|5.7|22.7|83.3|

<!-- Generated by ./scripts/utils/show_asr_result.sh -->
# asr_tamil
## Environments
- date: `Mon Jun 10 13:48:08 IST 2024`
- python version: `3.8.18 (default, Sep 11 2023, 13:40:15)  [GCC 11.2.0]`
- espnet version: `espnet 202308`
- pytorch version: `pytorch 1.13.1`
- Git hash: `f206a031225568000404105f119878818af9ce19`
  - Commit date: `Mon Jun 10 13:47:55 2024 +0530`
- ASR config: [conf/tuning/train_asr_transformer.yaml](conf/tuning/train_asr_transformer.yaml)
- Decode config: [conf/decode_asr.yaml](conf/decode_asr.yaml)
- nbpe: 700

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|2391|34963|70.0|26.0|4.0|5.4|35.4|87.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|2391|259927|91.7|4.0|4.3|5.0|13.4|87.2|

### TER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_asr_model_valid.acc.ave/eval|2391|109111|83.8|10.4|5.8|5.0|21.3|87.3|
