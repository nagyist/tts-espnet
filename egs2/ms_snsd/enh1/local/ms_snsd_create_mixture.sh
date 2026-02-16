#!/usr/bin/env bash

configure=   # Path to the configure file

. utils/parse_options.sh
. path.sh
. cmd.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 --configure <configure> <ms_snsd> <ms_snsd_wav> "
  echo " where <ms_snsd> is ms_snsd directory,"
  echo " <ms_snsd_wav> is wav generation space."
  exit 1;
fi

ms_snsd=$1
ms_snsd_wav=$2

# You can tune these later
total_hours_train=30 #1 # increase later
total_hours_test=1 #0.3 # increase if needed
snr_lower=0
snr_upper=40
total_snrlevels=5
audio_length=10
silence_length=0.2
sampling_rate=16000
audioformat="*.wav"

rm -r data/ 2>/dev/null || true
mkdir -p data/

mix_script=${ms_snsd}/noisyspeech_synthesizer.py
base_cfg=${configure:-${ms_snsd}/noisyspeech_synthesizer.cfg}

# Inputs (match your repo layout)
speech_dir_train=${ms_snsd}/clean_train
noise_dir_train=${ms_snsd}/noise_train
speech_dir_test=${ms_snsd}/clean_test
noise_dir_test=${ms_snsd}/noise_test

for p in "${mix_script}" "${base_cfg}"; do
  [ -f "${p}" ] || { echo "Error: not found: ${p}"; exit 1; }
done
for d in "${speech_dir_train}" "${noise_dir_train}" "${speech_dir_test}" "${noise_dir_test}"; do
  [ -d "${d}" ] || { echo "Error: not found dir: ${d}"; exit 1; }
done

mkdir -p "${ms_snsd_wav}"

make_cfg () {
  local dst_cfg=$1
  local speech_dir=$2
  local noise_dir=$3
  local total_hours=$4

  # MS-SNSD cfg only supports these keys (no output destination keys!)
  cat > "${dst_cfg}" << EOF
[noisy_speech]
sampling_rate: ${sampling_rate}
audioformat: ${audioformat}
audio_length: ${audio_length}
silence_length: ${silence_length}
total_hours: ${total_hours}
snr_lower: ${snr_lower}
snr_upper: ${snr_upper}
total_snrlevels: ${total_snrlevels}
noise_dir: ${noise_dir}
speech_dir: ${speech_dir}
noise_types_excluded: None
EOF
}

run_one () {
  local tag=$1              # train | test
  local speech_dir=$2
  local noise_dir=$3
  local total_hours=$4
  local out_root=$5         # ${ms_snsd_wav}/train or /test

  local workdir=${ms_snsd_wav}/work_${tag}
  rm -rf "${workdir}"
  mkdir -p "${workdir}"

  # Copy required python files into workdir so outputs are created under workdir
  cp "local/ms_snsd_noisyspeech_synthesizer.py" "${workdir}/noisyspeech_synthesizer.py"
  # MS-SNSD script imports audiolib.py (and possibly others). Copy at least audiolib.py.
  cp "${ms_snsd}/audiolib.py" "${workdir}/"

  local cfg=${workdir}/cfg.cfg
  make_cfg "${cfg}" "${speech_dir}" "${noise_dir}" "${total_hours}"

  (cd "${workdir}" && python noisyspeech_synthesizer.py --cfg cfg.cfg --cfg_str noisy_speech)

  # The script writes fixed folder names under workdir:
  #   NoisySpeech_training, CleanSpeech_training, Noise_training
  mkdir -p "${out_root}"
  rm -rf "${out_root}/noisy" "${out_root}/clean" "${out_root}/noise" 2>/dev/null || true
  mv "${workdir}/NoisySpeech_training" "${out_root}/noisy"
  mv "${workdir}/CleanSpeech_training" "${out_root}/clean"
  mv "${workdir}/Noise_training"       "${out_root}/noise"
}

echo "[Stage1] Generate TRAIN mixtures"
run_one train "${speech_dir_train}" "${noise_dir_train}" "${total_hours_train}" "${ms_snsd_wav}/train"

echo "[Stage1] Generate TEST mixtures"
run_one test  "${speech_dir_test}"  "${noise_dir_test}"  "${total_hours_test}"  "${ms_snsd_wav}/test"

# if [ -z "$configure" ]; then
#   # modify path in the original noisyspeech_synthesizer.cfg
#   configure=${ms_snsd}/noisyspeech_synthesizer.cfg
#   train_cfg=data/noisyspeech_synthesizer.cfg


#   if [ ! -f ${configure} ]; then
#     echo -e "Please check configurtion ${configure} exist"
#     exit 1;
#   fi

#   #input datas
#   noise_dir_train=${ms_snsd}/noise_train
#   speech_dir_train=${ms_snsd}/clean_train
#   noise_dir_test=${ms_snsd}/noise_test
#   speech_dir_test=${ms_snsd}/clean_test

#   #outputs
#   out_train=${ms_snsd_wav}/train
#   out_test=${ms_snsd_wav}/test
#   log_dir=data/log
#   mkdir -p "${out_train}" "${out_test}"

#   mix_script=${ms_snsd}/noisyspeech_synthesizer.py
#   base_cfg=${ms_snsd}/noisyspeech_synthesizer.cfg

#   if [ ! -f "${mix_script}" ]; then
#     echo "Error: ${mix_script} not found"
#     exit 1
#   fi
#   if [ -z "${configure}" ]; then
#     configure="${base_cfg}"
#   fi
#   if [ ! -f "${configure}" ]; then
#     echo "Error: cfg ${configure} not found"
#     exit 1
#   fi
#   # cfg copies
#   train_cfg=${ms_snsd_wav}/noisyspeech_synthesizer_train.cfg
#   test_cfg=${ms_snsd_wav}/noisyspeech_synthesizer_test.cfg

#   # ---- helper: generate cfg ----
#   make_cfg () {
#     local src_cfg=$1
#     local dst_cfg=$2
#     local speech_dir=$3
#     local noise_dir=$4
#     local out_root=$5
#     local total_hours=$6

#     local noisy_wav=${out_root}/noisy
#     local clean_wav=${out_root}/clean
#     local noise_wav=${out_root}/noise
#     mkdir -p "${noisy_wav}" "${clean_wav}" "${noise_wav}"

#     # NOTE: output key names MUST match MS-SNSD cfg.
#     sed -e "/^snr_lower/s#.*#snr_lower: ${snr_lower}#g" \
#         -e "/^snr_upper/s#.*#snr_upper: ${snr_upper}#g" \
#         -e "/^total_hours/s#.*#total_hours: ${total_hours}#g" \
#         -e "/^noise_dir/s#.*#noise_dir: ${noise_dir}#g" \
#         -e "/^speech_dir/s#.*#speech_dir: ${speech_dir}#g" \
#         -e "/^log_dir/s#.*#log_dir: ${log_dir}#g" \
#         -e "/^noisy_destination/s#.*#noisy_destination: ${noisy_wav}#g" \
#         -e "/^clean_destination/s#.*#clean_destination: ${clean_wav}#g" \
#         -e "/^noise_destination/s#.*#noise_destination: ${noise_wav}#g" \
#         "${src_cfg}" > "${dst_cfg}"
#   }

#   echo "[Stage1] Generate TRAIN mixtures"
#   make_cfg "${configure}" "${train_cfg}" "${speech_train}" "${noise_train}" "${out_train}" "${total_hours_train}"
#   python "${mix_script}" --cfg "${train_cfg}"

#   echo "[Stage1] Generate TEST mixtures"
#   make_cfg "${configure}" "${test_cfg}" "${speech_test}" "${noise_test}" "${out_test}" "${total_hours_test}"
#   python "${mix_script}" --cfg "${test_cfg}"
