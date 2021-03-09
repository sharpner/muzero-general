pip install -v -r requirements.txt
pip3 install -v -r requirements.txt
sleep 60 #wait till cuda is ready
CUDA_LAUNCH_BLOCKING=1 python3 muzero.py tictacnine
