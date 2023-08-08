close all;
warning('off');
matlab.video.read.UseHardwareAcceleration('off');  

%replace filepath with images folder
imagesFolder = 'C:\Users\Max\Documents\Y3 Uni\Computer Vision\ACW\testimages';

%replace filepath with path to video
videoPath = 'C:\Users\Max\Documents\Y3 Uni\Computer Vision\ACW\testvideos\shortClip.mp4';

%replace types with relevant image/video types
imagesType = "*.jpg";

testImages = imageFolderToStruct(imagesFolder, imagesType);

%go through images and retrieve signs
identifySignsInImages(testImages);

%go through videos and retreive signs
%identifySignsInVideo(videoPath);

%complete sign identification process on a array of images
function identifySignsInImages(testImages)
    numOfImages = size(testImages.image, 2);
    disp("number of images detected: " + numOfImages);

    z = 1;

    %go through images and retrieve signs
    for i = 1:numOfImages
        if z == 5
            figure;
            z = 1;
        end
        subplot(4,2,(z*2)-1);
        image = testImages.image{i};
        imshow(image);
        subplot(4,2,(z*2));

        %where the magic happens
        signs = signsIdentifier(image);

        disp("Image " + i + ": ");
        for s = 1:size(signs.isolatedWhite,2)
            signName = signs.templates.name(signs.templateMatches{s});
            signName = erase(signName{1},".png");
            %display information on detected signs
            if(signName ~= "NoMatch");
                disp("Sign Type: " + signName);
                disp("Sign Width: ");
                disp(fix(signs.filledboundingbox{s}(3)));
                disp("Sign Height: ");
                disp(fix(signs.filledboundingbox{s}(4)));
                disp("Sign Location: ");
                disp(fix(signs.filledcentroid{s}(1)));
                disp(fix(signs.filledcentroid{s}(2)));
            end
        end
         
        z = z + 1;
    end
end

function identifySignsInVideo(videoPath)
    video = VideoReader(videoPath);
    disp("video loaded");

    %the interval of frames to scan, increase number to scan less frames
    %but speed up overall process
    interval = 15;
    frames = {};
    f = 1;

    signs = struct;

    %open and begin writing to file
    fileID = fopen("VideoResults.txt","w");
    formatspectitle = "%s\nframe,timestamp,sign,x,y,width,height\n";
    fprintf(fileID,formatspectitle,video.Name);

    formatspecdata = "%i,%f,%s,%i,%i,%i,%i\n";

    while hasFrame(video)
        image = readFrame(video);
        if mod(f, interval) == 0
            frames{end + 1} = image;
        end
        f = f + 1;
    end
    disp("frames loaded");
    for i = 1:size(frames,2)
        signs = signsIdentifier(frames{i});
        disp("Image " + i + ": ");
        if size(signs.isolatedWhite,2) == 0
            disp("No signs detected");
        end
        for s = 1:size(signs.isolatedWhite,2);
            signName = signs.templates.name(signs.templateMatches{s});
            signName = erase(signName{1},".png");
            %display information on detected signs
            if ((signName{1} ~= "NoMatch") && (signs.filledarea{s} > 2600));
                disp("Frame: " + i*interval); 
                disp("Timestamp: " + video.CurrentTime);
                disp("Sign Type: " + signName);
                disp("Sign Bounding Box: ");
                disp(fix(signs.filledboundingbox{s}));
                disp("Sign Centroid: ");
                disp(fix(signs.filledcentroid{s}));
                %write data line to file
                dataToWrite = [i*interval,round(video.CurrentTime,1),fix(signs.filledcentroid{s}(1)),fix(signs.filledcentroid{s}(2)),fix(signs.filledboundingbox{s}(3)),fix(signs.filledboundingbox{s}(4))];
                disp(dataToWrite);
                fprintf(fileID,formatspecdata,dataToWrite);
            end
        end
    end
    fclose(fileID);
end

%compares two images of size 64x64
function result = signComparer(image1,image2)
    matches = 0;
    for x = 1:64
        for y = 1:64
            if image1(x,y) == image2(x,y)
                matches = matches + 1;
            end
        end
    end
    result = matches/4096;
end

function result = signsIdentifier(image)
    %uses a colour filter to find red signs
    filteredImageRed = combinedRedFilter(image);
    filteredImageBlue = combinedBlueFilter(image);
    filteredImage = filteredImageRed + filteredImageBlue;

    %remove elements that are too small to be valid signs via pure area removal
    %then filledarea removal
    morphedImage = bwareaopen(filteredImage,300,4); 
    morphedImage = bwpropfilt(morphedImage, "FilledArea", [400,1000000]);

    %identify connected objects and retrieve their properties
    objects = bwconncomp(morphedImage, 4);
    stats = regionprops("table",objects,"Centroid", ...
    "Area","BoundingBox","Circularity");

    %signs struct contains sign identification results
    signs = struct;
    signs.isolatedSigns = [];
    signs.isolatedWhite = [];

    signs.filledboundingbox = [];
    signs.filledcentroid = [];
    signs.isolatedWhiteArea = [];
    signs.filledArea = [];

    %loop through all the connected objects in the image
    for y = 1:objects.NumObjects
        %create a mask of the connected object and apply it to the original
        %image to retrieve the potential sign
        signMask = zeros(size(image));
        signMask(objects.PixelIdxList{y}) = 1;
        signMask = imbinarize(rgb2gray(signMask));
        signMask = imfill(signMask,"holes");

        %records areas and centroids of signs for reference once signs have
        %been processed
        filledstats = regionprops(signMask, "BoundingBox", "Centroid", "Area");
        signs.filledboundingbox{y} = filledstats.BoundingBox;
        signs.filledcentroid{y} = filledstats.Centroid;
        signs.filledarea{y} = filledstats.Area;

        %removes items with extremely low circularity, as these cannot be
        %signs
        tempstats = regionprops(signMask, "Circularity");
        if tempstats.Circularity < 0.32
            signMask = zeros(size(image));
            signMask = imbinarize(rgb2gray(signMask));
        end

        %morphology to make sign shapes more consistent
        signMask = imerode(signMask, strel("disk", 6));
        signMask = imdilate(signMask, strel("disk", 4));

        maskedRgbImage = bsxfun(@times, image, cast(signMask, 'like', image));
        signs.isolatedSigns{y} = maskedRgbImage;

        %use thresholding to seperate the white from the other parts of the
        %sign
        graySign = rgb2gray(maskedRgbImage);
        level = multithresh(graySign,2);
        whiteSegment = imquantize(graySign,level);
        labelledSegments = label2rgb(whiteSegment);
        yellowExtractionMask = (labelledSegments(:,:,1) > 0.2);
        isolatedYellow = bsxfun(@times, labelledSegments, cast(yellowExtractionMask, 'like', labelledSegments));
        isolatedWhite = imbinarize(rgb2gray(isolatedYellow));
        isolatedWhite = bwareaopen(isolatedWhite,50,4);

        %crops sign down to it's bounding box
        [rows, columns] = find(isolatedWhite);
        row1 = min(rows);
        row2 = max(rows);
        col1 = min(columns);
        col2 = max(columns);
        isolatedWhite = isolatedWhite(row1:row2, col1:col2); 

        if rows > 1
            isolatedWhite = imresize(isolatedWhite, [64,64]);
            signs.isolatedWhite{y} = isolatedWhite;
        else
            isolatedWhite = zeros(64,64);
            signs.isolatedWhite{y} = isolatedWhite;
        end
    end

    %retrieve the templates from their folder and put them in an array
    currentFile = matlab.desktop.editor.getActiveFilename;
    currentFolder = fileparts(currentFile);
    signs.templates = imageFolderToStruct(currentFolder + "\templates","*.png");
    numOfTemplates = size(signs.templates.image, 2);
    binarizedTemplates = [];
    
    for n = 1:numOfTemplates
        binarizedTemplates{n} = imbinarize(signs.templates.image{n});
    end

    %remove all blank inputs
    blank = imbinarize(generateBlankTemplate());
    for y = 1:size(signs.isolatedWhite,2);
        if signs.isolatedWhite{y} == blank
            signs.isolatedWhite{y} = [];
        end
    end

    signs.templateMatches = [];
    for x = 1:size(signs.isolatedWhite,2);
        %array that stores all the ratings for comparisons to the sign
        %templates
        ratings = struct;
        ratings.rating = [];
        ratings.sign = [];

        %loops through the templates and compares the white space in the
        %sign to them, producing a rating
        for n = 1:numOfTemplates
            %if isolatedWhite is empty, do not bother with template testing          
            if isempty(signs.isolatedWhite{x})
                ratings.rating{n} = 0;
                ratings.sign{n} = "NoMatch.png";
                break;
            end
            rating = signComparer(binarizedTemplates{n},signs.isolatedWhite{x});
            ratings.rating{n} = rating;
            ratings.sign{n} = signs.templates.name{n};
        end
        
        %if the highest rating is above 0.8, store the maximum rating and
        %the sign it corresponds to
        [bestRating, bestRatingIndex] = max([ratings.rating{:,:}]);
        bestSign = ratings.sign(bestRatingIndex);
        if bestRating > 0.82
            signs.templateMatches{x} = bestRatingIndex;            
        else
            %sets sign index to the no match template
            signs.templateMatches{x} = 15;
        end
    end

    signs.detectedSigns = [];

    %store detected signs in signs struct
    for k = 1:size(signs.templateMatches, 2)
        if signs.templateMatches{k} ~= 15
            signs.detectedSigns{end + 1} = signs.templates.image{signs.templateMatches{k}};
        end
    end

    %montage(signs.detectedSigns);

    result = signs;
end

function result = generateCircleTemplate()
    template = generateBlankTemplate();
    template = insertShape(template, "FilledCircle", [32,32,32], "Opacity", 1);
    result = imbinarize(rgb2gray(template));
end

function result = generateOctagonTemplate()
    template = generateBlankTemplate();
    template = insertShape(template, "FilledPolygon", [18,0,46,0, 64,18,64,46, 46,64,18,64,0,46,0,18]);
    result = imbinarize(rgb2gray(template));
end

function result = generateUpTriangleTemplate()
    template = generateBlankTemplate();
    template = insertShape(template, "FilledPolygon", [0,61,30,0,34,0,64,61,61,64,3,64]);
    result = imbinarize(rgb2gray(template));
end

function result = generateDownTriangleTemplate()
    template = generateBlankTemplate();
    template = insertShape(template, "FilledPolygon", [3,0,0,3,30,64,34,64,64,3,61,0]);
    result = imbinarize(rgb2gray(template));
end

function result = generateOneWayRectangleTemplate()
    template = generateBlankTemplate();
    template = insertShape(template, "FilledPolygon", [16,0,48,0,48,64,16,64]);
    result = imbinarize(rgb2gray(template));
end

%generates a blank 64x64 binary image
function result = generateBlankTemplate()
    array = zeros(64,64);
    result = mat2gray(array);
end

%makes a struct of all the images in a folder
function result = imageFolderToStruct(imagesFolder, type)
    imagesPattern = fullfile(imagesFolder, type);
    testImagesDirectory = dir(imagesPattern);
    for i = 1:length(testImagesDirectory)
        fullFileName = fullfile(imagesFolder, testImagesDirectory(i).name);
        testImages.image{i} = imread(fullFileName);
        testImages.name{i} = (erase(fullFileName,imagesFolder + "\"));
    end
    result = testImages;
end

%combines lab red filter with a hsv saturation filter and finally a rgb
%filter
function result = combinedRedFilter(image)
    hsvImage = rgb2hsv(image);
    labImage = rgb2lab(image);
    %main filter, excludes some signs so other filters are used and then
    %added

    labmask = ((labImage(:,:,2) > 35) & (labImage(:,:,3) > 5));
    labImage = bsxfun(@times, labImage, cast(labmask, 'like', labImage));
    labmaskedimage = lab2rgb(labImage);

    %mask for red signs in bright lighting
    hsvmask2 = ((hsvImage(:,:,1) > 0.009) & (hsvImage(:,:,1) < 0.052)) & (hsvImage(:,:,2) > 0.117) & (hsvImage(:,:,2) < 0.466) & (hsvImage(:,:,3) > 0.907);
    hsvFilteredImage2 = bsxfun(@times, hsvImage, cast(hsvmask2, 'like', hsvImage));
    hsvFilteredImage2 = hsv2rgb(hsvFilteredImage2);
    
    %mask for red signs in dark lighting
    hsvmask3 = (hsvImage(:,:,1) >= 0.951 ) & (hsvImage(:,:,1) <= 0.998) & ...
    (hsvImage(:,:,2) >= 0.592 ) & (hsvImage(:,:,2) <= 0.751) & ...
    (hsvImage(:,:,3) >= 0.309 ) & (hsvImage(:,:,3) <= 0.421);
    hsvFilteredImage3 = bsxfun(@times, hsvImage, cast(hsvmask3, 'like', hsvImage));
    hsvFilteredImage3 = hsv2rgb(hsvFilteredImage3);

    %mask for parts of signs where their reflection has made them appear
    %purple
    hsvmask4 = (hsvImage(:,:,1) >= 0.895 ) & (hsvImage(:,:,1) <= 0.956) & ...
    (hsvImage(:,:,2) >= 0.343 ) & (hsvImage(:,:,2) <= 0.789) & ...
    (hsvImage(:,:,3) >= 0.243 ) & (hsvImage(:,:,3) <= 0.585);
    hsvFilteredImage4 = bsxfun(@times, hsvImage, cast(hsvmask4, 'like', hsvImage));
    hsvFilteredImage4 = hsv2rgb(hsvFilteredImage4);

    %mask for blurred edge of stop sign
    hsvMask5 = (hsvImage(:,:,1) >= 0.987 ) | (hsvImage(:,:,1) <= 0.011) & ...
    (hsvImage(:,:,2) >= 0.494 ) & (hsvImage(:,:,2) <= 0.662) & ...
    (hsvImage(:,:,3) >= 0.509 ) & (hsvImage(:,:,3) <= 0.655);
    hsvFilteredImage5 = bsxfun(@times, hsvImage, cast(hsvMask5, 'like', hsvImage));
    hsvFilteredImage5 = hsv2rgb(hsvFilteredImage5);

    combinedImage = labmaskedimage + hsvFilteredImage3 + hsvFilteredImage2 + hsvFilteredImage4 + hsvFilteredImage5;
    grayCombinedImage = rgb2gray(combinedImage);
    binarizedImage = imbinarize(grayCombinedImage);
    result = binarizedImage;
end

function result = combinedBlueFilter(image)
    hsvImage = rgb2hsv(image);
    labImage = rgb2lab(image);

    hsvmask1 = (hsvImage(:,:,1) >= 0.537 ) & (hsvImage(:,:,1) <= 0.584) & ...
    (hsvImage(:,:,2) >= 0.438 ) & (hsvImage(:,:,2) <= 1) & ...
    (hsvImage(:,:,3) >= 0.392 ) & (hsvImage(:,:,3) <= 0.934);
    hsvFilteredImage1 = bsxfun(@times, hsvImage, cast(hsvmask1, 'like', hsvImage));
    hsvFilteredImage1 = hsv2rgb(hsvFilteredImage1);

    hsvmask2 = (hsvImage(:,:,1) >= 0.572 ) & (hsvImage(:,:,1) <= 0.638) & ...
    (hsvImage(:,:,2) >= 0.528 ) & (hsvImage(:,:,2) <= 0.791) & ...
    (hsvImage(:,:,3) >= 0.449 ) & (hsvImage(:,:,3) <= 0.796);
    hsvFilteredImage2 = bsxfun(@times, hsvImage, cast(hsvmask2, 'like', hsvImage));
    hsvFilteredImage2 = hsv2rgb(hsvFilteredImage2);

    hsvmask3 = (hsvImage(:,:,1) >= 0.56 ) & (hsvImage(:,:,1) <= 0.619) & ...
    (hsvImage(:,:,2) >= 0.242 ) & (hsvImage(:,:,2) <= 1) & ...
    (hsvImage(:,:,3) >= 0.0 ) & (hsvImage(:,:,3) <= 0.577);
    hsvFilteredImage3 = bsxfun(@times, hsvImage, cast(hsvmask3, 'like', hsvImage));
    hsvFilteredImage3 = hsv2rgb(hsvFilteredImage3);

    hsvmask4 = (hsvImage(:,:,1) >= 0.594 ) & (hsvImage(:,:,1) <= 0.65) & ...
    (hsvImage(:,:,2) >= 0.468 ) & (hsvImage(:,:,2) <= 1) & ...
    (hsvImage(:,:,3) >= 0.408 ) & (hsvImage(:,:,3) <= 0.800);
    hsvFilteredImage4 = bsxfun(@times, hsvImage, cast(hsvmask4, 'like', hsvImage));
    hsvFilteredImage4 = hsv2rgb(hsvFilteredImage4);

    hsvmask5 = (hsvImage(:,:,1) >= 0.585 ) & (hsvImage(:,:,1) <= 0.639) & ...
    (hsvImage(:,:,2) >= 0.347 ) & (hsvImage(:,:,2) <= 1) & ...
    (hsvImage(:,:,3) >= 0 ) & (hsvImage(:,:,3) <= 0.664);
    hsvFilteredImage5 = bsxfun(@times, hsvImage, cast(hsvmask5, 'like', hsvImage));
    hsvFilteredImage5 = hsv2rgb(hsvFilteredImage5);

    combinedImage = hsvFilteredImage1 + hsvFilteredImage2 + hsvFilteredImage3 + hsvFilteredImage4 + hsvFilteredImage5;
    grayCombinedImage = rgb2gray(combinedImage);
    binarizedImage = imbinarize(grayCombinedImage);
    result = binarizedImage;
end