#NoteScanner

###Introduction

The goal of this project is to make an tool to digitize hand-written notes. Taking notes by hand is often easier and faster than typing up notes, especially for technical material. Nonetheless, having a digital copy of your notes is helpful both as a backup and as a quick resource (you can't `grep` over a notebook). There are tools that do this natively (namely [LiveScribe](www.livescribe.com/smartpen/)), but they are significantly more expensive than ordinary pens and paper.

The process of digitizing notes will go something like this:

1. Take a picture of your notes.
2. Use some basic Computer Vision to recognize the sheet of paper and find its corners (hopefully 4).
3. Transform the image so the identified corners are dragged to the corners of the image
4. Recognize handwriting.

Full documentation is available on [Github Pages](http://ben-eysenbach.github.io/NoteScanner/) (which supports MathJax).
