kaggle datasets download -d borhanitrash/cat-dataset -p data
unzip -q data/cat-dataset.zip -d data
mv data/cats data/cats1
echo "Dataset1 download and unzip done."

if test -f data/uardzbqun8t47olxygkgel.zip; then
    unzip -q data/uardzbqun8t47olxygkgel.zip -d data
    mv data/images.cv_uardzbqun8t47olxygkgel data/cats2
    echo "Dataset2 unzip done"
else
    echo "Please download the second dataset from https://images.cv/dataset/cat-image-classification-dataset with 64x64 size and paste to the data folder before."
fi