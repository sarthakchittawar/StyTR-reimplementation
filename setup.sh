mkdir /scratch/sarthak
wait
scp sarthak.chittawar@ada.iiit.ac.in:/share1/sanika/wikiart.zip /scratch/sarthak &
scp sarthak.chittawar@ada.iiit.ac.in:/share1/dataset/coco/train2017.tar /scratch/sarthak &
scp sarthak.chittawar@ada.iiit.ac.in:/share1/dataset/coco/val2017.tar /scratch/sarthak &
scp sarthak.chittawar@ada.iiit.ac.in:/share1/dataset/coco/test2017.tar /scratch/sarthak &
wait
unzip /scratch/sarthak/wikiart
tar -xvf /scratch/sarthak/train2017.tar
tar -xvf /scratch/sarthak/val2017.tar
tar -xvf /scratch/sarthak/test2017.tar