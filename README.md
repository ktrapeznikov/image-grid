# image-grid
Just make me an image grid 

```python
im = make_grid_from_files(files,texts="index")
im.save("grid.jpg")
```

or

```bash
python image_grid.py "test_images/*.png" --output-file "grid.jpg" --texts "index"
```


![Image Grid](grid.jpg)



```bash
python image_grid.py --help
Usage: image_grid.py [OPTIONS] FILES

Options:
  --aspect FLOAT
  --size INTEGER
  --texts TEXT
  --text-color TEXT
  --output-file TEXT
  --help              Show this message and exit.

```
