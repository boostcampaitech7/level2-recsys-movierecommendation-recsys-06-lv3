source .env

ssh server$server_no << 'ENDSSH'
if pgrep -f "/opt/conda/" > /dev/null
then
    echo "app.py가 실행 중입니다. 종료를 시작합니다."

    # '/opt/conda/bin/python3'가 포함된 프로세스 종료
    pids=$(pgrep -f '/opt/conda/')

    if [ -z "$pids" ]; then
        echo "No processes found with /opt/conda/bin/python3."
    else
        for pid in $pids; do
            echo "Killing process ID: $pid"
            kill -9 $pid  # -9 옵션은 강제 종료를 의미합니다.
        done
        echo "All specified processes have been killed."
    fi

    if [ $? -eq 0 ]; then
        echo "app.py가 성공적으로 종료되었습니다."
    else
        echo "app.py 종료에 실패했습니다."
    fi
else
    echo "app.py가 실행 중이 아닙니다."
fi
ENDSSH
read