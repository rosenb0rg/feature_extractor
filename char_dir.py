import os

class Character:
	"""
	a "character" class that lets you store all the location of subfiles for the conversion
	process and quickly pull specific
	"""
	def __init__(self, name, scene, source, target):
		base = 'D:/'
		# judgeC
		self.name = name
		# raupach
		self.scene = scene
		# 003
		self.source = source
		# 03b
		self.target = target
		# raupach_judgeC_s003_t03b
		self.full_name = scene + "_" + name +"_s" + str(source) + "_t" + str(target)
		# D:/characters/judgeC/
		
		self.basedir = os.path.abspath(base + 'characters/%s/' % name)
		# D:/characters/judgeC/face/
		self.facedir = os.path.abspath(self.basedir + '/face/')
		# D:/characters/judgeC/src/align
		self.aligndir = os.path.abspath(self.basedir + '/src/align/')
		# D:/characters/judgeC/src/comp
		self.compdir = os.path.abspath(self.basedir + '/src/comp/')
		# D:/source/raupach/judgeC/
		self.srcdir = os.path.abspath(base + '/source/' + '%s/%s/' % (scene, name))
		
		# subdirectories for workflow elements
		self.align_png_dir = os.path.abspath(self.aligndir + '/%s/png' % self.full_name)
		self.align_crop_dir = os.path.abspath(self.aligndir + '/%s/crop' % self.full_name)
		self.align_conv_dir = os.path.abspath(self.aligndir + '/%s/converted' % self.full_name)
		self.align_comped_dir = os.path.abspath(self.aligndir + '/%s/comped' % self.full_name)
		self.align_obj_dir = os.path.abspath(self.aligndir + '/%s/obj' % self.full_name)

		# subdirectories for faceswap_mod process
		self.swap_head_dir = os.path.abspath(self.facedir + "/" + name + "_t" + str(target))
		self.swap_comp_dir = os.path.abspath(self.compdir + "/" + self.full_name)

		# subdirectories for deepfakes
		self.model_dir = os.path.abspath(self.basedir + "/df/model_GAN128")
		self.imgA_dir = os.path.abspath(base + 'source/00_training/crop/')
		self.imgB_dir = os.path.abspath(self.basedir + "/df/imgB")
		self.imgB_crop_dir = os.path.abspath(self.basedir + "/df/imgB_crop")

		# self.src_dict()
		# self.face_dict()
		# self.align_dict()

	# def src_dict(self):
	# 	src_dict = {}
	# 	for root, dirs, files in os.walk(self.srcdir):
	# 		for name in dirs:
	# 			dir = (os.path.join(root, name))
	# 			src_num = dir.split("_")[-1]
	# 			src_dict[src_num] = []
	# 			for filename in os.listdir(dir):
	# 				if filename.endswith('.jpg'):
	# 					src_dict[src_num].append(filename)
	# 	self.src_dict = src_dict

	# def face_dict(self):
	# 	face_dict = {}
	# 	for root, dirs, files in os.walk(self.facedir):
	# 		for name in dirs:
	# 			dir = (os.path.join(root, name))
	# 			face_num = dir.split("_")[1]
	# 			face_dict[face_num] = []
	# 			for filename in os.listdir(dir):
	# 				if filename.endswith('.jpg') or filename.endswith('.png'):
	# 					face_dict[face_num].append(os.path.join(root, filename))
	# 	self.face_dict = face_dict

	# def align_dict(self):
	# 	align_dict = {}
	# 	for root, dirs, files in os.walk(self.aligndir):
	# 		align_dict[self.aligndir] = []
	# 		for dir in dirs:
	# 			sub_dir = str(os.path.join(root, dir))
	# 			key = sub_dir.split("_")[-2] + sub_dir.split("_")[-1]
	# 			align_dict[key] = (sub_dir)
	# 		del dirs[:]
	# 	self.align_dict = align_dict

	# def align_png_dir(self):
	# 	temp = 's' + str(self.source) + 't' + str(self.target)
	# 	align_png_dir = self.align_dict[temp] + '/png'
	# 	align_png_dir = os.path.abspath(align_png_dir)
	# 	#print ('aligned pngs', align_png_dir)
	# 	return align_png_dir

	# def align_crop_dir(self):
	# 	temp = 's' + str(self.source) + 't' + str(self.target)
	# 	align_crop_dir = self.align_dict[temp] + '/crop'
	# 	align_crop_dir = os.path.abspath(align_crop_dir)
	# 	#print ('crop dir', align_crop_dir)
	# 	return align_crop_dir
	
	# def align_conv_dir(self):
	# 	temp = 's' + str(self.source) + 't' + str(self.target)
	# 	align_conv_dir = self.align_dict[temp] + '/converted'
	# 	align_conv_dir = os.path.abspath(align_conv_dir)
	# 	#print ('converted dir', align_conv_dir)
	# 	return align_conv_dir

	# def align_comped_dir(self):
	# 	temp = 's' + str(self.source) + 't' + str(self.target)
	# 	align_comped_dir = self.align_dict[temp] + '/comped'
	# 	align_comped_dir = os.path.abspath(align_comped_dir)
	# 	#print ('recomposited faces dir', align_comped_dir)
	# 	return align_comped_dir

if __name__ == "__main__":
	x = Character('judgeC', 'raupach', '001', '00b')
	# print (x.name)
	# print (x.basedir)
	# print (x.facedir)
	# print (x.aligndir)
	# print (x.compdir)
	# print (x.srcdir)
	print (x.swap_comp_dir)