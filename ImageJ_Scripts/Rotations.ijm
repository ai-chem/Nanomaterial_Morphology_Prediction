input = "G:/Nanomaterial_Morphology_Prediction/Datasets/Augmented_Image_Dataset/";
output = "G:/Nanomaterial_Morphology_Prediction/Datasets/Final_Image_Dataset/Version_0/";

// This script creates a copy of each image with different 45 degree rotations, 8 for each image

function action(input, output, filename, rotation) {

        // Open each image and save rotated image

        open(input + filename);
		name = filename;
		ang = rotation * 45 - 90;
		run("Rotate... ", "angle=ang grid=1 interpolation=Bilinear fill");
		name = "angle" + rotation + "_" + name;
        saveAs("Jpeg", output + name);
        close();
}

setBatchMode(true); 
list = getFileList(input);
for (i = 0; i < list.length; i++){
	for (k = 0; k < 8; k++){
		action(input, output, list[i], k);
	}	
}
setBatchMode(false);
