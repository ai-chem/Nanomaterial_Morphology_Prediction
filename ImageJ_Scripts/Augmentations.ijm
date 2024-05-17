input = "G:/Nanomaterial_Morphology_Prediction/Datasets/Segmented_Image_Dataset/";
output = "G:/Nanomaterial_Morphology_Prediction/Datasets/Augmented_Image_Dataset/";

// This script is used to create two additional copies of each image by changing it's brightness and sharpness

function action(input, output, filename, brightness, sharpness) {

        // Open each image and produce bright (dim) or blurred (sharpened) copy

        open(input + filename);
		name = filename;
		if (brightness == 1) {
		  setMinAndMax(25, 300);
		  name = "dim_" + name;
		}
		if (brightness == 0) {
		  setMinAndMax(0, 150);
		  name = "bright_" + name;
		}
		if (sharpness == 0) {
		  run("Smooth");
		  name = "smooth_" + name;
		}
		if (sharpness == 1) {
		  run("Sharpen");
		  name = "sharp_" + name;
		}

        // Save each image

        saveAs("Jpeg", output + name);
        close();
}

setBatchMode(true); 
list = getFileList(input);
for (i = 0; i < list.length; i++){
	for (k = 0; k < 3; k++){
		for (m = 0; m < 3; m++){
			action(input, output, list[i], m, k);
		}
	}	
}
setBatchMode(false);
