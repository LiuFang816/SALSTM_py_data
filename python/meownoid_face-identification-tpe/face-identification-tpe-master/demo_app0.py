from model import FaceVerificator
from skimage import io

###
img_path_0 = '/home/egor/1.jpg'
img_path_1 = '/home/egor/2.jpg'
dist = 0.85
###

fv = FaceVerificator('./model')
fv.initialize_model()

img_0 = io.imread(img_path_0)
img_1 = io.imread(img_path_1)

faces_0 = fv.process_image(img_0)
faces_1 = fv.process_image(img_1)

n_faces_0 = len(faces_0)
n_faces_1 = len(faces_1)

if n_faces_0 == 0 or n_faces_1 == 0:
    print('Error: No faces found on the {}!'.format(img_path_0 if n_faces_0 == 0 else img_path_1))
    exit()

rects_0 = list(map(lambda p: p[0], faces_0))
rects_1 = list(map(lambda p: p[0], faces_1))

embs_0 = list(map(lambda p: p[1], faces_0))
embs_1 = list(map(lambda p: p[1], faces_1))

scores, comps = fv.compare_many(dist, embs_0, embs_1)

print('Rects on image 0: {}'.format(rects_0))
print('Rects on image 1: {}'.format(rects_1))

# print('Embeddings of faces on image 0:')
# print(embs_0)
#
# print('Embeddings of faces on image 1:')
# print(embs_1)

print('Score matrix:')
print(scores)

print('Decision matrix :')
print(comps)
