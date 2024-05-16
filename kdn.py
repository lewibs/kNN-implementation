import torch
from knn import generate_points, get_neighbors

#KDN stands for k distance neighbor. Rather then x neighbors, its neighbors in d 

def euclidean_distance(a,b):
    return torch.sqrt(torch.sum((a - b) ** 2))

def get_neighbors(train, test_row, distance):
    distances = [(train_row, euclidean_distance(test_row[:-1], train_row[:-1])) for train_row in train]
    
    neighbors = []
    for point, d in distances:
        if d.item() < distance:
            neighbors.append(point)

    return torch.stack(neighbors)


def generate_points(n, d):
    groups = torch.rand(n)
    groups *= 2
    groups = torch.floor(groups)
    points = torch.rand(n, d)
    points = points[torch.argsort(points[:, 0])]
    groups = groups[torch.argsort(groups)]
    groups = groups.unsqueeze(1)
    points = torch.cat((points, groups), dim=1)
    return points

def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)[1:] #remove the current one
    predictions = [int(row[-1].item()) for row in neighbors]
    prediction = max(set(predictions), key=predictions.count)
    return prediction

if __name__ == "__main__":
    points = generate_points(1000,3)
    prediction = predict_classification(points, points[800], 0.1)
    print('Expected %d, Got %d.' % (points[800][-1], prediction))