conda env create -f environment.yml

cd src/lib/models/networks/
rm -rf DCNv2
git clone https://github.com/CharlesShang/DCNv2.git
cd DCNv2
./make.sh
conda activate