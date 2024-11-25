source .env
scp -r $data_path server$server_no:/data/ephemeral/home/data/data.csv
ssh server$server_no << ENDSSH
cd /data/ephemeral/home/data
ls -lta
ENDSSH
./3.start-app.sh