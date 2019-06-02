if [ -d "bilstm-aux" ]; then
  echo 'bilstm-aux is already installed. skipping.'
  exit
fi

git clone https://github.com/bplank/bilstm-aux.git