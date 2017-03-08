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
        if get_sorted:
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
    suggestions_unsorted = label_suggestions(images, image_labels, centroids, labels, losses, get_sorted=False)
    
    #print(suggestions)
    def classifier(image, display_best=False, metric=get_distance, breakdown=False):
        vimage = image.reshape(-1)
        
        distances = []
        
        least_index = None
        least_distance = None
        uninitialized = True
        
        #print("Leastloss:", losses[-1])
        
        for i, centroid in enumerate(centroids):
            vcentroid = centroid.reshape(-1)
            distance = metric(vimage, vcentroid)
            distances += [[i, distance]]
            
        
        distances = sorted(distances, key=lambda x: x[1])
        
        
        if breakdown:
            for d in distances:
                #print(d)
                pass
        
        #print(distances[0][0])
        least_index = distances[0][0]
        
#        print(suggestions)
        
        corr = []
        
        cql = [[x, 0] for x in range(len(distances))]
        #print(cql)
        for i, dt in enumerate(distances):
            idx = dt[0]
            d = dt[1]
            # distance ratio, lower values is closer
            dr = d / losses[-1]
            dinv = 1 / dr
            #print(idx, "->", suggestions[idx])
            qlist = [(q[0], int(100 * q[1] * dinv)/100) for q in suggestions_unsorted[idx]]
            for j, qz in enumerate(qlist):
                cql[j][1] += qz[1]
            #print("Distance:", dinv, " digits:", qlist)

        cql = sorted(cql, key=lambda x: -x[1])
        #print(cql)
        
        maxed_digit = suggestions[least_index][0][0]
        
        if display_best:
            display_arr(centroids[least_index])
        
        THE_METHOD = 1
        
        digit = None
        
        if THE_METHOD == 1:
            digit = maxed_digit
        elif THE_METHOD == 2:
            digit = cql[0][0]
                
        
        
        return digit
    return classifier

def testzzz(idx, bundle, images, labels):
    classifier = classifier_factory(A_images, A_labels, bundle['centroids'], bundle['labels'], bundle['losses'])
    label = int(labels[idx][0])
    display_arr(images[idx])
    print("Actual:", label)
    digit = classifier(images[idx], display_best=True, breakdown=True)
    print("Got:", digit)



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
    
    # raða og minnka
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