import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def create_font(fontname='Tahoma', fontsize=10):
	return { 'fontname': fontname, 'fontsize':fontsize }

def subplot(title, images, rows, cols, sptitle="subplot", sptitles=[], colormap=cm.gray, ticks_visible=True, filename=None):
	fig = plt.figure()
	# main title
	fig.text(.5, .95, title, horizontalalignment='center') 
	for i in xrange(len(images)):
		ax0 = fig.add_subplot(rows,cols,(i+1))
		plt.setp(ax0.get_xticklabels(), visible=False)
		plt.setp(ax0.get_yticklabels(), visible=False)
		if len(sptitles) == len(images):
			plt.title("%s #%s" % (sptitle, str(sptitles[i])), create_font('Tahoma',10))
		else:
			plt.title("%s #%d" % (sptitle, (i+1)), create_font('Tahoma',10))
		plt.imshow(np.asarray(images[i]), cmap=colormap)
	if filename is None:
		plt.show()
	else:
		fig.savefig(filename)
		
def imsave(image, title="", filename=None):
	plt.figure()
	plt.imshow(np.asarray(image))
	plt.title(title, create_font('Tahoma',10))
	if filename is None:
		plt.show()
	else:
		fig.savefig(filename)
