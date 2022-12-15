import cv2
import numpy as np

def sift_detector(new_image, image_template):
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    # image1 = cv2.Canny(image1, 300, 550, None, 3)
    image2 = cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.Canny(image2, 150, 300, None, 3)

    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)
    print("Keypoints found in image: ", len(descriptors_1))
    print("Keypoints found in template: ", len(descriptors_2))

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 1)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # the result 'matchs' is the number of similar matches found in both images
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    return matchesMask, image1, image2, keypoints_1, keypoints_2, matches

def sift_detectorBF(new_image, image_template):
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    # image1 = cv2.Canny(image1, 350, 500, None, 3)
    image2 = cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.Canny(image2, 300, 500, None, 3)

    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)
    print("Keypoints found in image: ", len(descriptors_1))
    print("Keypoints found in template: ", len(descriptors_2))

    bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=False)
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    print("Matches: ", len(matches))

    return image1, image2, keypoints_1, keypoints_2, matches


def ORB_detector(new_image, image_template):

    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    # image1 = cv2.Canny(image1, 350, 500, None, 3)
    image2 = cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.Canny(image2, 300, 500, None, 3)

    orb = cv2.ORB_create()

    keypoints_1, descriptors_1 = orb.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image2, None)
    print("Keypoints found in image: ", len(descriptors_1))
    print("Keypoints found in template: ", len(descriptors_2))
    
    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # the result 'matchs' is the number of similar matches found in both images
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    return matchesMask, image1, image2, keypoints_1, keypoints_2, matches

def ORB_detectorBF(new_image, image_template):

    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    # image1 = cv2.Canny(image1, 350, 500, None, 3)
    image2 = cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.Canny(image2, 300, 500, None, 3)

    orb = cv2.ORB_create()

    keypoints_1, descriptors_1 = orb.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image2, None)
    print("Keypoints found in image: ", len(descriptors_1))
    print("Keypoints found in template: ", len(descriptors_2))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    print("Matches: ", len(matches))

    return image1, image2, keypoints_1, keypoints_2, matches

def BRIEF_detector(new_image, image_template):
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    # image1 = cv2.Canny(image1, 350, 500, None, 3)
    image2 = cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.Canny(image2, 300, 500, None, 3)

    # Initiate FAST detector
    star = cv2.xfeatures2d.StarDetector_create()
    # Initiate BRIEF extractor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    # find the keypoints with STAR
    keypoints_1 = star.detect(image1, None)
    # compute the descriptors with BRIEF
    keypoints_1, descriptors_1 = brief.compute(image1, keypoints_1)
    keypoints_2 = star.detect(image2, None)
    # compute the descriptors with BRIEF
    keypoints_2, descriptors_2 = brief.compute(image2, keypoints_2)
    print("Keypoints found in image: ", len(descriptors_1))
    print("Keypoints found in template: ", len(descriptors_2))

    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # the result 'matchs' is the number of similar matches found in both images
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)


    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    return matchesMask, image1, image2, keypoints_1, keypoints_2, matches

def BRIEF_detectorBF(new_image, image_template):
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    # image1 = cv2.Canny(image1, 350, 500, None, 3)
    image2 = cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.Canny(image2, 300, 500, None, 3)

    # Initiate FAST detector
    star = cv2.xfeatures2d.StarDetector_create()
    # Initiate BRIEF extractor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    # find the keypoints with STAR
    keypoints_1 = star.detect(image1, None)
    # compute the descriptors with BRIEF
    keypoints_1, descriptors_1 = brief.compute(image1, keypoints_1)
    keypoints_2 = star.detect(image2, None)
    # compute the descriptors with BRIEF
    keypoints_2, descriptors_2 = brief.compute(image2, keypoints_2)
    print("Keypoints found in image: ", len(descriptors_1))
    print("Keypoints found in template: ", len(descriptors_2))

    bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    print("Matches: ", len(matches))

    return image1, image2, keypoints_1, keypoints_2, matches

def Surf_detector(new_image, image_template):
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    # image1 = cv2.Canny(image1, 350, 500, None, 3)
    image2 = cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.Canny(image2, 300, 500, None, 3)

    surf = cv2.xfeatures2d.SURF_create()

    keypoints_1, descriptors_1 = surf.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = surf.detectAndCompute(image2, None)
    print("Keypoints found in image: ", len(descriptors_1))
    print("Keypoints found in template: ", len(descriptors_2))

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=1)
    search_params = dict(checks=50)

    # Create the Flann Matcher object
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    return matchesMask, image1, image2, keypoints_1, keypoints_2, matches

def Surf_detectorBF(new_image, image_template):
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    # image1 = cv2.Canny(image1, 350, 500, None, 3)
    image2 = cv2.cvtColor(image_template, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.Canny(image2, 300, 500, None, 3)

    surf = cv2.xfeatures2d.SURF_create()

    keypoints_1, descriptors_1 = surf.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = surf.detectAndCompute(image2, None)
    print("Keypoints found in image: ", len(descriptors_1))
    print("Keypoints found in template: ", len(descriptors_2))

    bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    print("Matches: ", len(matches))

    return image1, image2, keypoints_1, keypoints_2, matches
