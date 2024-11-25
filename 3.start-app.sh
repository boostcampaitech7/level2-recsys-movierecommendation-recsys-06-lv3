#!/bin/bash

source .env
scp -r config.yaml server$server_no:/data/ephemeral/home/$git_project_name

# ssh 명령어에 .env에서 불러온 변수 사용
ssh server$server_no << ENDSSH
#!/bin/bash
cd /data/ephemeral/home/level2-competitiveds-recsys-06

git fetch
git pull

/opt/conda/bin/python3 -m pip install -r ./requirements.txt

cd ./src
# app.py가 실행 중인지 확인
if pgrep -f "/opt/conda/bin/python3 app.py" > /dev/null
then
    echo "app.py가 이미 실행 중입니다. 실행을 패스합니다."
else
    echo "app.py가 실행 중이 아닙니다. 실행을 시작합니다."

    # 원격 서버에서 현재 시간 구하기
    now=\$(date +"%Y%m%d-%H%M%S")

    # 로그 파일 생성 및 app.py 실행
    nohup /opt/conda/bin/python3 app.py > "$data_log_path"/app-\$now.log 2>&1 &
    echo "app.py가 백그라운드에서 실행되었습니다. 로그는 "$data_log_path"/app-\$now.log에 기록됩니다."
fi
ENDSSH

read