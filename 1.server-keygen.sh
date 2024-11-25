#!/bin/bash
source .env

ssh server$server_no << ENDSSH
mkdir "$data_log_path"
mkdir -p /data/ephemeral/home/data/output
git config --global user.name "$username"
git config --global user.email "$email"
git config --global --list

ssh-keygen -t ed25519 -C "$email" -N "" -f /root/.ssh/id_ed25519

echo "SSH 키가 생성되었습니다. 다음 명령어로 공개 키를 복사 후 Git 개인 SSH에 등록하세요(Auth&Sign 둘다):"
cat /root/.ssh/id_ed25519.pub
ENDSSH

echo "모든 작업이 완료되었습니다."
read


