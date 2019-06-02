if [ -d "ud-treebanks-v2.4" ]; then
  echo 'ud-treebanks-v2.4 is already downloaded. skipping.'
  exit
fi

curl https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2988/ud-treebanks-v2.4.tgz | tar xvz ud-treebanks-v2.4/UD_English-ParTUT ud-treebanks-v2.4/UD_English-EWT