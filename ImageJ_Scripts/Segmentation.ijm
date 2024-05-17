// This script is used for segmentation of each individual nanoparticle image

for ( j=0; j<=214; j++ ) {

    // Open each initial SEM image and select each individual particle

    open("G:/Nanomaterial_Morphology_Prediction/Datasets/Initial_Image_Dataset/"+j+".tif");
    orig = getTitle();
    setBatchMode(true);
    run("8-bit");
    run("Duplicate...", "title=cpy");
    run("Subtract Background...", "rolling=60 sliding disable");
    setAutoThreshold("Triangle dark");
    setOption("BlackBackground", true);
    run("Convert to Mask");
    run("Fill Holes");
    
    // We will only consider particles from a specific size range, as very small or large particles are more likely to be outliers
    // We will also only work with particles that are not connected to the edges of an image for their shape to be more presice

    run("Analyze Particles...", "size=250-8000 exclude add");
    
    // Select original image to work with each particle selection in original image

    selectImage(orig);
    roiManager("Show All");
    close("cpy");
    n = roiManager("count");
    for ( i=0; i<n; i++ ) {

       // For each particle in an image create an empy black image and copy particle to this image

       selectImage(orig);
       run("Duplicate...", "title=cpy");
       roiManager("select", i);
       run("Make Inverse");
       setBackgroundColor(0, 0, 0);
       run("Clear", "slice");
       run("Make Inverse");
       run("Crop");
       newImage(j+"_"+i, "RGB black", 224, 224, 1);
       current = getTitle();
       run("Select All");
       run("Image to Selection...", "image=cpy opacity=100 zero");
       run("Flatten");
       run("Select All");

       // Save individual image of a particle on a black background and close all images

       saveAs("JPG", "G:/Nanomaterial_Morphology_Prediction/Datasets/Segmented_Image_Dataset/"+j+"_"+i+".jpg");
       close();
       close();
       close();
       }
    roiManager("reset");
    close("ROI Manager");
    run("Select None");
    setBatchMode("exit");
    close();
}
