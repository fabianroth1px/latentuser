import tensorflow as tf
import numpy as np
import random
import glob
import collections
import json
import sets
import uuid

USER_DIM = 8
ITEM_DIM = 8
HIDDEN = 8
ETA = 0.05
ETA_W = 0.01
NOISE = 0.01
STEPS_MINI_OPT = 50
#STEPS_MINI_OPT = 10
STEPS_MINI_OPT_W = 100
#RESTRICT_ITEMS_PER_USER = 0
#RESTRICT_USERS_PER_ITEM = 0
RESTRICT_ITEMS_PER_USER = 15
RESTRICT_USERS_PER_ITEM = 30

class Colearner:
	def __init__(self, model_name, sess, user_dim, item_dim, hidden_units, eta, eta_w, steps_mini_opt, steps_mini_opt_w, noise):
		self.model_name = model_name
		self.sess = sess
		self.user_dim = user_dim
		self.item_dim = item_dim
		self.hidden_units = hidden_units
		self.eta = eta
		self.eta_w = eta_w
		self.steps_mini_opt = steps_mini_opt
		self.steps_mini_opt_w = steps_mini_opt_w
		self.noise = noise

		self.buildNetwork()

		self.labels = tf.placeholder(tf.float32, [None, 1], "labels")
		self.loss = tf.reduce_mean(tf.square(self.out - self.labels))

		self.user_grad = tf.reduce_mean(tf.gradients(self.loss, self.f_u), 1)
		self.item_grad = tf.reduce_mean(tf.gradients(self.loss, self.f_i), 1)
		self.w_grad_optimizer = tf.train.GradientDescentOptimizer(eta_w).minimize(self.loss)

		self.sess.run(tf.initialize_all_variables())
		self.saver = tf.train.Saver()

	def dumpModel(self):
		self.saver.save(self.sess, "%s.ckpt" % self.model_name)
			

	def buildNetwork(self):
		with tf.name_scope("JointUserItemP") as scope:
			self.f_u = tf.placeholder(tf.float32, [None, self.user_dim], "f_u")
			self.f_i = tf.placeholder(tf.float32, [None, self.item_dim], "f_i")

			W_u = tf.Variable(tf.truncated_normal([self.user_dim, self.hidden_units]))
			W_i = tf.Variable(tf.truncated_normal([self.item_dim, self.hidden_units]))
			b_h = tf.Variable(tf.truncated_normal([self.hidden_units]))

			self.hidden = tf.tanh(tf.matmul(self.f_u, W_u) + tf.matmul(self.f_i, W_i) + b_h)

#			W_h2 = tf.Variable(tf.truncated_normal([self.hidden_units, self.hidden_units]))
#			b_h2 = tf.Variable(tf.truncated_normal([self.hidden_units]))
#
#			self.hidden2 = tf.tanh(tf.matmul(self.hidden, W_h2) + b_h2)

			W_o = tf.Variable(tf.truncated_normal([self.hidden_units, 1]))
			b_o = tf.Variable(tf.truncated_normal([1], 0, 1e-02))

			self.out = tf.tanh(tf.matmul(self.hidden, W_o) + b_o)

	def trainItemProb(self, f_i, f_us, out):
		lpre=self.sess.run(self.loss, feed_dict = {
			self.f_i: f_i, self.f_u: f_us, self.labels: out})

		for n in range(self.steps_mini_opt):
			if n % 10:
				print n
			curr_f_i = f_i

			grad_f_i = self.sess.run(self.item_grad, feed_dict = {
				self.f_i: curr_f_i, self.f_u: f_us, self.labels: out})
			curr_f_i = curr_f_i - self.eta * grad_f_i

		lpost=self.sess.run(self.loss, feed_dict = {
			self.f_i: curr_f_i, self.f_u: f_us, self.labels: out})

		return curr_f_i, lpre, lpost

	def getAffinity(self, f_us, f_is):
		out = self.sess.run(self.out, feed_dict = {
			self.f_i: f_is, self.f_u: f_us})
		return out

	def getTopN(self, f_u, f_is, N):
		out = self.sess.run(self.out, feed_dict = {
			self.f_i: f_is, self.f_u: f_u})
		sort_idx = sorted(enumerate(out), key=lambda x:-x[1])
		return sort_idx[0:N-1]


	def trainItem(self, f_i, f_us, out): 
		if np.random.uniform(0, 1) < self.noise:
			curr_f_i = np.random.randn(1, self.item_dim)
		else:
			curr_f_i = f_i
		lpre=self.sess.run(self.loss, feed_dict = {
			self.f_i: curr_f_i, self.f_u: f_us, self.labels: out})
		for n in range(self.steps_mini_opt):
			grad_f_i = self.sess.run(self.item_grad, feed_dict = {
				self.f_i: curr_f_i, self.f_u: f_us, self.labels: out})
			curr_f_i = curr_f_i - self.eta * grad_f_i

		lpost=self.sess.run(self.loss, feed_dict = {
			self.f_i: curr_f_i, self.f_u: f_us, self.labels: out})

		return curr_f_i, lpre, lpost

	def trainUser(self, f_u, f_is, out, debug): 
		if np.random.uniform(0, 1) < self.noise:
			curr_f_u = np.random.randn(1, self.user_dim)
		else:
			curr_f_u = f_u
		lpre=self.sess.run(self.loss, feed_dict = {
			self.f_u: curr_f_u, self.f_i: f_is, self.labels: out})
		for n in range(self.steps_mini_opt):
			grad_f_u = self.sess.run(self.user_grad, feed_dict = {
				self.f_u: curr_f_u, self.f_i: f_is, self.labels: out})
			curr_f_u = curr_f_u - self.eta * grad_f_u

		lpost=self.sess.run(self.loss, feed_dict = {
			self.f_u: curr_f_u, self.f_i: f_is, self.labels: out})
		if debug:
			print "f_u"
			print f_u
			print "f_is"
			print f_is
			print "out"
			print out
			print "lout"
			lout = self.sess.run(self.out, feed_dict = {
				self.f_u: curr_f_u, self.f_i: f_is, self.labels: out})
			print lout
			print "loss"
			print lpost


		return curr_f_u, lpre, lpost

	def trainW(self, f_us, f_is, out):
		lpre=self.sess.run(self.loss, feed_dict = {
			self.f_u: f_us, self.f_i: f_is, self.labels: out})
		for n in range(self.steps_mini_opt_w):
			self.sess.run(self.w_grad_optimizer, feed_dict = {
				self.f_u: f_us, self.f_i: f_is, self.labels: out})
		lpost=self.sess.run(self.loss, feed_dict = {
			self.f_u: f_us, self.f_i: f_is, self.labels: out})
		return lpre, lpost

Item = collections.namedtuple('Item', ['id', 'f_i', 'users'])
User = collections.namedtuple('User', ['id', 'f_u', 'items'])

class UserItemStore:
	def __init__(self, fileglob, user_dim, item_dim, N_users, N_items):
		self.user_dim = user_dim
		self.item_dim = item_dim
		
		self.users = {}
		self.items = {}

		self.loadData(fileglob)
		self.restrictData(N_users, N_items)

		self.items_set = sets.Set(self.items.keys())
		self.users_set = sets.Set(self.users.keys())

	def showSomeUsers(self):
		n = 20
		for i,t in self.users.iteritems():
			if len(t.items) == 1:
				continue
			print "%s %s" % (i,t)
			n -= 1
			if n == 0:
				break

	def showSomeItems(self):
		n = 20
		for i,t in self.items.iteritems():
			if len(t.users) == 1:
				continue
			print "%s %s" % (i,t)
			n -= 1
			if n == 0:
				break

	def evalSet(self, fileglob):
		files = glob.glob(fileglob)
		f_us = []
		f_is = []
		for file in files:
			f = open(file, 'r')
			for line in f:
				e = line.rstrip().split(",")
				if len(e) != 2:
					print "bad format: %s" % line
					continue
				user, item = e
				if user not in self.users:
					continue
				if item not in self.items:
					continue
				f_us.append(self.users[user].f_u)
				f_is.append(self.items[item].f_i)

		return (np.asmatrix(np.asarray(f_us)), np.asmatrix(np.asarray(f_is)))


	def loadData(self, fileglob):
		files = glob.glob(fileglob)
		print files
		for file in files:
			f = open(file, 'r')
			for line in f:
				e = line.rstrip().split(",")
				if len(e) != 2:
					print "bad format: %s" % line
					continue
				user, item = e
				if user == "658885":
					print "skipping bad user 658885"
					continue
				if user in self.users:
					user_entry = self.users.get(user)
					user_entry.items.add(item)
				else:
					f_u = np.random.randn(1, self.user_dim)
					user_entry = User(id=user, f_u=f_u, items=sets.Set([item]))
					self.users[user] = user_entry

				if item in self.items:
					item_entry = self.items.get(item)
					item_entry.users.add(user)
				else:
					f_i = np.random.randn(1, self.item_dim)
					item_entry = Item(id=item, f_i=f_i, users=sets.Set([user]))
					self.items[item] = item_entry

		print 'users: %d' % len(self.users)
		print 'items: %d ' % len(self.items)

	def lookupItem(self, id):
		return self.items[id].f_i

	def lookupUser(self, id):
		return self.users[id].f_u
		
	def restrictData(self, N_users, N_items):
		# uset, iset: users/items with large amounts of items/users.
		uset = sets.Set()
		for id, u in self.users.iteritems():
			if len(u.items) > N_items:
				uset.add(id)
		iset = sets.Set()
		for id, i in self.items.iteritems():
			if len(i.users) > N_users:
				iset.add(id)

		users = {}
		for id, u in self.users.iteritems():
			if id not in uset:
				continue
			u = u._replace(items=u.items.intersection(iset))
			if len(u.items) > 0:
				users[id] = u
		items = {}
		for id, i in self.items.iteritems():
			if id not in iset:
				continue
			i = i._replace(users = i.users.intersection(uset))
			if len(i.users) > 0:
				items[id] = i

		self.users = users
		self.items = items
		print 'users: %d' % len(self.users)
		print 'items: %d ' % len(self.items)


	# iterate over users and return matrix of items/non-items and labels
	def getUsersItems(self):
		for _, u in self.users.iteritems():
			n_items = len(u.items)
			n_neg = n_items
			d = np.zeros([n_items + n_neg, self.item_dim])
			o = np.ones([n_items + n_neg, 1])
			for i,item in enumerate(u.items):
				f_i = self.items[item].f_i
				d[i,:] = f_i.flatten()
			non_items = self.items_set.difference(u.items)
			if len(non_items) < n_items:
				print "[%s] [%s] [%s]" % (u.id, u.items, non_items)
				n_items = len(non_items)
			neg_samples = random.sample(non_items, n_neg)
			for i,s in enumerate(neg_samples):
				d[n_items + i, :] = self.items[s].f_i
				o[n_items + i, 0] = -1.0
			yield (u.id, u.f_u, d, o)

	# iterate over items and return matrix of users/non-users and labels
	def getItemsUsers(self):
		for _, i in self.items.iteritems():
			n_users = len(i.users)
			n_neg = n_users
			d = np.zeros([n_users + n_neg, self.user_dim])
			o = np.ones([n_users + n_neg, 1])
			for j,user in enumerate(i.users):
				f_u = self.users[user].f_u
				d[j,:] = f_u.flatten()
			non_users = self.users_set.difference(i.users)
			neg_samples = random.sample(non_users, n_neg)
			for j,s in enumerate(neg_samples):
				d[n_users + j, :] = self.users[s].f_u
				o[n_users + j, 0] = -1.0
			yield (i.id, i.f_i, d, o)

	def getItemsUsersBatch(self):
		n_samples = 0
		for _, i in self.items.iteritems():
			n_samples += len(i.users)

		f_us = np.zeros([2 * n_samples, self.user_dim])
		f_is = np.zeros([2 * n_samples, self.item_dim])
		o = np.ones([2 * n_samples, 1])
		j = 0
		for _, i in self.items.iteritems():
			n_users = len(i.users)
			n_neg = n_users / 2
			for _,user in enumerate(i.users):
#				f_us[j, :] = self.users[user].f_u.flatten()
				f_us[j, :] = self.users[user].f_u
				f_is[j, :] = i.f_i
				j+=1
			non_users = self.users_set.difference(i.users)
			neg_samples = random.sample(non_users, n_neg)
			for _,s in enumerate(neg_samples):
				f_us[j, :] = self.users[s].f_u
				f_is[j, :] = i.f_i
				o[j, 0] = -1.0
				j+=1

		return f_us, f_is, o

	def getAllItems(self):
		ids = []
		f_is = []
		for id,item in self.items.iteritems():
			ids.append(id)
			f_is.append(item.f_i)
		return (ids, np.asmatrix(np.asarray(f_is)))

	def getAllUsers(self):
		ids = []
		f_us = []
		for id,user in self.users.iteritems():
			ids.append(id)
			f_us.append(user.f_u)
		return (ids, np.asmatrix(np.asarray(f_us)))


	def getUser(self, id):
		return self.users[id]

	def getItem(self, id):
		return self.items[id]

	def getUsers(self):
		for _, u in self.users.iteritems():
			n_items = len(u.items)
			d = np.zeros([n_items, self.user_dim + self.item_dim])
			for i,item in enumerate(u.items):
				f_i = self.items[item].f_i
				d[i,:] = np.concatenate((u.f_u, f_i)).flatten()
			yield (u.id, d)

	def getItems(self):
		for _, i in self.items.iteritems():
			n_items = len(i.users)
			d = np.zeros([n_items, self.user_dim + self.item_dim])
			for j,user in enumerate(i.users):
				f_u = self.users[user].f_u
				d[i,:] = np.concatenate((f_u, i.f_i))
			yield (i.id, d)

	def setUserFu(self, id, f_u):
		self.users[id] = self.users[id]._replace(f_u=f_u)

	def setItemFi(self, id, f_i):
		self.items[id] = self.items[id]._replace(f_i=f_i)

	def exportAsDict(self):
		items = []
		for _, i in self.items.iteritems():
			if len(i.users) > 0:
				items.append({
					"id": i.id,
					"v": i.f_i.flatten().tolist(),
					"len": len(i.users)
					})
		users = []
		for _, u in self.users.iteritems():
			if len(u.items) > 0:
				users.append({
					"id": ",".join(u.items),
					"v": u.f_u.flatten().tolist(),
					"len": len(u.items)
					})
		j = {
			"items": items,
			"users": users
		}
		return j


#us = UserItemStore("/home/ubuntu/data/part*", USER_DIM, ITEM_DIM, RESTRICT_USERS_PER_ITEM, RESTRICT_ITEMS_PER_USER)
us = UserItemStore("../../data/part-000[0-8]*", USER_DIM, ITEM_DIM, RESTRICT_USERS_PER_ITEM, RESTRICT_ITEMS_PER_USER)
#us = UserItemStore("testdata/*", USER_DIM, ITEM_DIM, RESTRICT_USERS_PER_ITEM, RESTRICT_ITEMS_PER_USER)
us.showSomeUsers()

model_name = uuid.uuid1()
json_filename = "%s.json" % model_name

print "Model name: %s" % model_name

sess = tf.InteractiveSession()
cl = Colearner(model_name, sess, USER_DIM, ITEM_DIM, HIDDEN, ETA, ETA_W, STEPS_MINI_OPT, STEPS_MINI_OPT_W, NOISE)

def exportJson(json_filename, j, step):
	j["meta"] = {
		"user_dim": USER_DIM,
		"item_dim": ITEM_DIM,
		"hidden": HIDDEN,
		"eta": ETA,
		"eta_W": ETA_W,
		"steps_mini_opt": STEPS_MINI_OPT,
		"steps_mini_opt_w": STEPS_MINI_OPT_W,
		"noise": NOISE,
		"step": step
	}
	f = open(json_filename, 'w')
	json.dump(j, f, indent=2)
	f.close()

j = us.exportAsDict()
exportJson(json_filename, j, 0)

(users,f_us) = us.getAllUsers()
users = np.random.permutation(users)

for step in range(1000000):
	(items,f_is) = us.getAllItems()

	for i in range(5):
		user = us.getUser(users[i])
		print user
		topN = cl.getTopN(user.f_u, f_is, 40)
		topItems = [(items[i],s) for i,s in topN]
		print ",".join(("%f: %s" % (s,id) for id, s in topItems))
	
	llpre = 0
	llpost = 0
	n = 0
	for iid, f_i, f_us, out in us.getItemsUsers():
		f_i_out, lpre, lpost = cl.trainItem(f_i, f_us, out)
		us.setItemFi(iid, f_i_out) 
		llpre = llpre + lpre
		llpost = llpost + lpost
		n += 1
	print "Items %d: pre: %f, post: %f" % (step, llpre / n, llpost / n)
	llpre = 0
	llpost = 0
	n = 0
	for uid, f_u, f_is, out in us.getUsersItems():
		if step % 10 == 0:
			debug = False
		else:
			debug = False
		f_u_out, lpre, lpost = cl.trainUser(f_u, f_is, out, debug)
		us.setUserFu(uid, f_u_out) 
		llpre = llpre + lpre
		llpost = llpost + lpost
		n += 1
	print "Users %d: pre: %f, post: %f" % (step, llpre / n, llpost / n)
	j = us.exportAsDict()
	
	f_us, f_is, o = us.getItemsUsersBatch()
	lpre, lpost = cl.trainW(f_us, f_is, o)
	print "W %d: pre: %f, post: %f" % (step, lpre, lpost)

	fu_eval, fi_eval = us.evalSet("../../data/part-0000[0-3]")
	out = cl.getAffinity(fu_eval, fi_eval)
	out2 = cl.getAffinity(fu_eval, np.random.permutation(fi_eval))
	print "Training: on %f, roll %f" % (np.mean(out), np.mean(out2))
	fu_eval, fi_eval = us.evalSet("../../data/part-0009[0-3]")
	out = cl.getAffinity(fu_eval, fi_eval)
	out2 = cl.getAffinity(fu_eval, np.random.permutation(fi_eval))
	print "Eval: on %f, roll %f" % (np.mean(out), np.mean(out2))

	exportJson(json_filename, j, step+1)
	cl.dumpModel()

