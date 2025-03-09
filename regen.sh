#!/bin/bash
set -ex

git fetch
git co origin/main

if [ ! -e llvm-project ]; then
  git clone https://github.com/llvm/llvm-project
  mkdir llvm-project/build
  ( cd llvm-project/build && cmake ../llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON  -GNinja )
else
  ( cd llvm-project && git pull --ff-only )
fi
HASH=`cd llvm-project && git rev-parse HEAD`
( cd llvm-project/build && ninja )

PATH=$PWD/llvm-project/build/bin:$PATH python generate.py

if ! git diff-index --quiet HEAD --; then
  git add -u
  git commit -m "Regen $HASH `date`"
  git push origin HEAD:main
fi
