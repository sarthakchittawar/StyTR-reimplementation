mkdir /scratch/sarthak
cd /scratch/sarthak
wait
scp sarthak.chittawar@ada.iiit.ac.in:/share1/sanika/wikiart.zip . &
scp sarthak.chittawar@ada.iiit.ac.in:/share1/dataset/coco/train2017.tar . &
scp sarthak.chittawar@ada.iiit.ac.in:/share1/dataset/coco/val2017.tar . &
scp sarthak.chittawar@ada.iiit.ac.in:/share1/dataset/coco/test2017.tar . &
wait
unzip wikiart
wait
python ~/StyTR-reimplementation/split.py
tar -xvf train2017.tar
tar -xvf val2017.tar
tar -xvf test2017.tar
