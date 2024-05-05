# Reimplementation of Sty-TR 
(style transfer using transformers)
<!-- add link -->
[Original Paper](https://arxiv.org/pdf/2105.14576) \
[Original Codebase](https://github.com/diyiiyiii/StyTR-2)

```
contributors:
Sarthak Chittawar
Sanika Damle
```

## Training
`python3 train.py` + additional arguments as specified in the file

## Using the model with CLIP
`python3 style_with_clip.py` + change file paths in the file

## Calculating loss if content, style and final image given
`python3 loss.py` + change file paths in the file

## Directory Structure
```
├── README.md
├── transformer files
├── clip
│   ├── finetuneclip.py (finetune clip)
│   ├── metadata.json (metadata of all the wikiart images)
│   ├── styles.txt (different styles)
│   ├── styles_desc.txt (description of styles)
├── content (content images)
├── style (style images)
├── output (output images)
├── videos
│   ├── video_Stabilisztion.ipynb (python notebook with all the relevant code)
│   ├── video files
```
