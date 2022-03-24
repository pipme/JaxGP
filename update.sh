#!/bin/zsh
html_dir=$(pwd)
docs_dir="../jaxgp/docs"
setopt extendedglob
rm -rf -- ^update.sh

cd $docs_dir
make ${1:-html}
echo $(pwd)
cd $html_dir
echo $(pwd)
cp -r $docs_dir/_build/html/* .
