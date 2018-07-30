import os

class Character:
	"""
	a "character" class that lets you store all the location of subfiles for the conversion
	process and quickly pull specific
	"""
	def __init__(self, name, scene, source, target):
		base = 'D:/'
		self.name = name
		self.scene = scene
		self.source = source
		self.target = target
		self.basedir = base + 'characters/%s/' % name
		self.facedir = self.basedir + 'face/'
		self.aligndir = self.basedir + 'src/align/'
		self.compdir = self.basedir + 'src/comp/'
		self.srcdir = base + 'source/' + '%s/%s/' % (scene, name)
		self.src_dict()
		self.face_dict()
		self.align_dict()

	def src_dict(self):
		src_dict = {}
		for root, dirs, files in os.walk(self.srcdir):
			for name in dirs:
				dir = (os.path.join(root, name))
				src_num = dir.split("_")[-1]
				src_dict[src_num] = []
				for filename in os.listdir(dir):
					if filename.endswith('.jpg'):
						src_dict[src_num].append(filename)
		self.src_dict = src_dict

	def face_dict(self):
		face_dict = {}
		for root, dirs, files in os.walk(self.facedir):
			for name in dirs:
				dir = (os.path.join(root, name))
				face_num = dir.split("_")[1]
				face_dict[face_num] = []
				for filename in os.listdir(dir):
					if filename.endswith('.jpg') or filename.endswith('.png'):
						face_dict[face_num].append(os.path.join(root, filename))
		self.face_dict = face_dict

	def align_dict(self):
		align_dict = {}
		for root, dirs, files in os.walk(self.aligndir):
			align_dict[self.aligndir] = []
			for dir in dirs:
				sub_dir = str(os.path.join(root, dir))
				key = sub_dir.split("_")[-2] + sub_dir.split("_")[-1]
				align_dict[key] = (sub_dir)
			del dirs[:]
		self.align_dict = align_dict

	def align_png_dir(self):
		temp = 's' + str(self.source) + 't' + str(self.target)
		align_png_dir = self.align_dict[temp] + '/png'
		align_png_dir = os.path.abspath(align_png_dir)
		#print ('aligned pngs', align_png_dir)
		return align_png_dir

	def align_crop_dir(self):
		temp = 's' + str(self.source) + 't' + str(self.target)
		align_crop_dir = self.align_dict[temp] + '/crop'
		align_crop_dir = os.path.abspath(align_crop_dir)
		#print ('crop dir', align_crop_dir)
		return align_crop_dir
	
	def align_conv_dir(self):
		temp = 's' + str(self.source) + 't' + str(self.target)
		align_conv_dir = self.align_dict[temp] + '/converted'
		align_conv_dir = os.path.abspath(align_conv_dir)
		#print ('converted dir', align_conv_dir)
		return align_conv_dir

	def align_comped_dir(self):
		temp = 's' + str(self.source) + 't' + str(self.target)
		align_comped_dir = self.align_dict[temp] + '/comped'
		align_comped_dir = os.path.abspath(align_comped_dir)
		#print ('recomposited faces dir', align_comped_dir)
		return align_comped_dir

# if __name__ == "__main__":
# 	x = Character('judgeC', 'raupach', '001', '00b')
# 	# print (x.name)
# 	# print (x.basedir)
# 	# print (x.facedir)
# 	# print (x.aligndir)
# 	# print (x.compdir)
# 	# print (x.srcdir)
# 	#print (x.align_dict)
# 	x.align_png_dir()
# 	x.align_crop_dir()
# 	x.align_comped_dir()
