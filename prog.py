
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage # notum þetta fyrir myndir

import datetime # Notað til að mæla tímamismun.
import kmeans # Staðbundin skrá, fyrir k-means
import math # Stærðfræði
import glob # Notað til að finna skrár.

import os

################################################################################
# Innlestur og teikning.


# Les inn mynd með SciPy og skilar henni sem
# 3D NumPy fylki.
# TODO: spurning hvort það eigi að vera dtype='uint8' hér
def load_image_as_arr(path):
    img = scipy.ndimage.imread(path)
    arr = np.asarray(img, dtype='uint8')
    return arr




# Birtir NumPY fylkið `arr' sem mynd.
def display_arr(arr, cast_to_uint8=True):
    if cast_to_uint8:
        plt.imshow(np.uint8(arr), cmap=plt.get_cmap('gray'), interpolation='nearest', vmin=0,vmax=255)
    else:
        plt.imshow(arr, cmap=plt.get_cmap('gray'), interpolation='nearest', vmin=0,vmax=255)
    plt.show()

################################################################################
# k-means meðhöndlun


# k-means fyrir fyrri hlutan.  Tekur inn eina albúmmynd.
def get_kmeans_1(arr, k=1):
    data = arr.reshape(-1, arr.shape[-1]).transpose()
    c, l, J = kmeans.kmeans(data, k)
    c2 = c.transpose()
    if False:
        # Svo hægt sé að teikna beint (óþarfi) en
        # gott að vita.
        c2 = np.reshape(c2, (1,) + c2.shape)
    return (c2, l, J)

def get_kmeans_2_forward_reshape(arr):
    return arr.reshape(arr.shape[0], -1).transpose()

# k-means fyrir seinni hlutan.  Tekur inn myndasafnið.
def get_kmeans_2(arr, k=1):
    data = arr.reshape(arr.shape[0], -1).transpose()
    c, l, J = kmeans.kmeans(data, k)
    c2 = c.transpose()
    c2 = c2.reshape(c2.shape[0], *arr.shape[1:])
    return (c2, l, J)



################################################################################
# Meðhöndlun á Bundle

def create_bundle(arr, k, kmeans_function, signature):
    duration = datetime.datetime.now()
    centroids, labels, losses = kmeans_function(arr, k)
    duration = datetime.datetime.now() - duration
    result = {
        "signature": signature, 
        "duration": str(duration),
        "k": k,
        "centroids": centroids,
        "labels": labels,
        "losses": losses
    }
    return result
    

def save_bundle(path, bundle):
    if path[-1] != '/':
        path += '/'
    the_date = datetime.datetime.now()
    the_date_string = the_date.strftime("%y-%m-%d-%H%M%S.%f")
    filename = path + "bundle-" + the_date_string + ".npy"
    payload = np.array({
        "filename": filename,
        "dateString": the_date_string,
        "bundle": bundle
    })
    np.save(filename, payload)

    
def load_bundle(path):
    return np.load(path)


def get_best_bundle(paths, signature, do_filter=False):
    least_loss = -1
    least_bundle = None
    least_path = None
    uninitialized = True
    s_paths = []
    for path in paths:
        if os.path.exists(path):
            arr = np.load(path)
            bundle = arr.item()['bundle']
            bundle_signature = bundle['signature']
            losses = bundle['losses']
            if signature == bundle_signature:
                s_paths += [path]
                loss = losses[-1]
                if uninitialized or loss < least_loss:
                    least_loss = loss
                    least_bundle = bundle
                    least_path = path
                    uninitialized = False
    
    if (least_path is not None) and (len(s_paths) >= 2) and (do_filter):
        s_paths.remove(least_path)
        for path in s_paths:
            # Final check
            if os.path.exists(path):
                os.remove(path)
    
    return least_bundle


def test_bundle(bundle, images, image_labels):
    losses = bundle['losses']
    centroids = bundle['centroids']
    labels = bundle['labels']
    plt.plot(range(0, len(losses)), losses)
    f2(images, image_labels, centroids, labels, losses)
    print("--- --- --- ---")

################################################################################
# Myndagreining

def f2(images, image_labels, centroids, labels, losses):
    stat = []
    for _ in centroids:
        sub = []
        for i in range(len(centroids)):
            sub += [[i, 0]]
        stat += [sub]
    #print(stat)
    
    for i, label in enumerate(labels):
        value = int(image_labels[i])
        stat[label][value][1] += 1
        #print(label, value)
        
    for i, it in enumerate(stat):
        n = sum([x[1] for x in it])
        #print(n)
        stat[i] = sorted(it, key=lambda x: -x[1])
        for j in range(len(stat[i])):
            stat[i][j][1] /= n
        #print(stat[i])
    
    for i, it in enumerate(centroids):
        display_arr(centroids[i])
        print("Centroid", i + 1, "/", len(centroids), "is")
        for j, it2 in enumerate(stat[i]):
            print("  ", it2[0], " with a prob. of", int(100 * it2[1]), "%")
            if j > 1:
                break
        print()

################################################################################
# Talnagreining

# stat[centroid][top_result][digit, ratio]
def label_suggestions(images, image_labels, centroids, labels, losses, get_sorted=True):
    stat = []
    for _ in centroids:
        sub = []
        for i in range(len(centroids)):
            sub += [[i, 0]]
        stat += [sub]
    
    for i, label in enumerate(labels):
        value = int(image_labels[i])
        stat[label][value][1] += 1
        #print(label, value)
        
    for i, it in enumerate(stat):
        n = sum([x[1] for x in it])
        stat[i] = sorted(it, key=lambda x: -x[1])
        for j in range(len(stat[i])):
            stat[i][j][1] /= n
            
    return stat



# x and y are vectors
def get_distance(x, y):
    u = x - y
    return np.dot(u,u)


def classifier_factory(images, image_labels, centroids, labels, losses):
    suggestions = label_suggestions(images, image_labels, centroids, labels, losses)
    
    #print(suggestions)
    def classifier(image, display_best=False, metric=get_distance, breakdown=False):
        vimage = image.reshape(-1)
        
        distances = []
        
        for i, centroid in enumerate(centroids):
            vcentroid = centroid.reshape(-1)
            distance = metric(vimage, vcentroid)
            distances += [[i, distance]]
            
        
        distances = sorted(distances, key=lambda x: x[1])
        
        least_index = distances[0][0]
        maxed_digit = suggestions[least_index][0][0]
        
        digit = maxed_digit
        
        return digit
    return classifier


################################################################################
# Búa til gögn


def generate_data_part1(albums, out_dir):
    if not os.path.isdir(out_dir):
        raise FileNotFoundError('Directory does not exist:', out_dir)
    print("Hluti 1")
    for i, album in enumerate(albums):
        print("Mynd", i + 1, "af", len(albums))
        bundle = create_bundle(album, 3, get_kmeans_1, str(i))
        save_bundle(out_dir, bundle)
        print("    Tími:", bundle["duration"])
    print()


def generate_data_part2(images, out_dir):
    if not os.path.isdir(out_dir):
        raise FileNotFoundError('Directory does not exist:', out_dir)
    # Hluti 2
    print("Hluti 2")
    for i, k in enumerate([10, 20, 30]):
        print("k =", k, ",", i + 1, "af", 3)
        bundle = create_bundle(images, k, get_kmeans_2, str(k))
        save_bundle(out_dir, bundle)
        print("    Tími:", bundle["duration"])
    print()


def save_bundle(path, bundle):
    if path[-1] != '/':
        path += '/'
    if not os.path.isdir(path):
        raise FileNotFoundError('Directory does not exist:', out_dir)
    the_date = datetime.datetime.now()
    the_date_string = the_date.strftime("%y-%m-%d-%H%M%S.%f")
    filename = path + "bundle-" + the_date_string + ".npy"
    payload = np.array({
        "filename": filename,
        "dateString": the_date_string,
        "bundle": bundle
    })
    np.save(filename, payload)


################################################################################
# DRASL

def gzf(arr, k):
    
    # Sýna myndina
    plt.imshow(arr, interpolation='nearest')
    plt.show()
    
    if True:
        return
    c, l, J = get_kmeans_2(arr, k)

    # Sýna fjölda ítranna
    print("ítranir", len(losses))
    print("Best", losses[-1])

    # Teikna fjölda ítrana
    plt.plot(range(0, len(losses)), losses)
    plt.show()

################################################################################


    