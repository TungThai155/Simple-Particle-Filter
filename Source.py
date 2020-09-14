########################################################################################################
# Created by Tung Thai                                                                                 #
# HW 2                                                                                                 #
# For COMP150 - PRP                                                                                    #
# Due Date March 04                                                                                    #
########################################################################################################
## standard import
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
from PIL import Image
#np.random.seed(123) #this will help to convert at exactly 70 moves
########################################################################################################
########################################### function ###################################################
########################################################################################################
# get width and height of the image
def get_pixels(filepath):
    im = Image.open(filepath)
    width, height = im.size #image size
    return width, height
########################################################################################################
# Set up data   
image_name = 'BayMap.png'
cwidth, cheight = get_pixels(image_name)
if cwidth > 2000:
    M = 50 #image obs and ref pixel
    print("Observation image size is %d" % M)
    NUM_PARTICLE = 100
elif cwidth > 1300 and cwidth < 2000:
    M = 30
    print("Observation image size is %d" % M)
    NUM_PARTICLE = 100
else:
    M = 40
    print("Observation image size is %d" % M)
    NUM_PARTICLE = 100
########################################################################################################
## check if out of bound:
def check_out_of_bound(x, x_bound, y, y_bound, image_size):
    if x < image_size//2 + 1 or x > x_bound - image_size//2 - 1 or y < 1 + image_size//2 or y > y_bound - image_size//2 - 1:
        #print("Out of bound")
        a = 1
    else:
        a = 0
    return a
########################################################################################################
def rmse(obs_img, ref_img):
    rms = np.sqrt(np.mean(np.square(obs_img-ref_img), axis=(0, 1)))
    return rms 
########################################################################################################
## Stochastic universal sampling
def SUS(population, n): # n is number of good offsprings
    # total fitness
    sum_fit = np.sum(population)
    p = sum_fit/n
    # start point anything from 0 to p
    start = np.random.uniform(high=p)
    pts= np.array([start + i*p for i in range(n)])
    return RWS(population, pts)
# fitness proportionate selection
def RWS(population, pts): #roulette wheel selection
    choose_idx = np.empty_like(pts, dtype=int) # index of keep particle
    for idx, p in enumerate(pts):
        i = 0
        fsp= population[0]
        while fsp < p:
            i = i + 1
            fsp = fsp + population[i]
        choose_idx[idx] = i
    return choose_idx #keep particle
########################################################################################################
## simple rank base fitness resampling particle filter
def RPF(particle, weight, width, height):
    #initialization
    sorted_weight = np.zeros((NUM_PARTICLE,1))
    prob = np.zeros(NUM_PARTICLE)
    #sort according to weight
    sorted_weight_idx = np.argsort(weight.T)  # assending order idx
    sorted_weight = np.sort(weight.T) #assending order weight
    sum_weight = np.sum(sorted_weight) # sum of weight
    for i in range(0,np.shape(particle)[0]):
        prob[i] = sorted_weight.flatten()[i]/sum_weight #normalize all the distribution
    sorted_weight_idx =  np.random.choice(sorted_weight_idx.flatten() , NUM_PARTICLE, p=prob.flatten())
    return sorted_weight_idx
########################################################################################################
## distance
def dist(x,x1,y,y1):
    distance = np.sqrt((x1-x)**2+(y1-y)**2)
    return distance
########################################################################################################
#############################################  class  ##################################################
########################################################################################################
#Control class
class Control:
    def __init__(self, imgdata, width, height):
        self.imgdata = imgdata
        # width, height
        self.width = width
        self.height = height

        # plot image and adjusted coordinate
        plt.figure()
        self.img = plt.imshow(imgdata)
        self.img.axes.set_xlim(0, self.width)
        self.img.axes.set_ylim(self.height, 0)
        self.img.axes.set_xticklabels((self.img.axes.get_xticks() - self.width // 2))
        self.img.axes.set_yticklabels((self.img.axes.get_yticks() - self.height // 2))

        # generate noise
        self.noise = 0
        self.x = np.random.uniform(M//2 + self.noise, self.width - M//2 - self.noise)
        self.y = np.random.uniform(M//2 + self.noise, self.height - M//2 - self.noise)

        # plot drone and particle
        self.drone = Drone(imgdata, self.width, self.height, self.x , self.y )
        self.par, self.dr = self.drone.plot(self.img)
        # handler
        self.cidpress = None
        # count move
        self.count = 0

    def connect(self):
        self.cidpress = self.img.figure.canvas.mpl_connect('key_press_event', self.on_press)

    def on_press(self, event):
        # remove
        if self.par is not None:
            self.par.remove()
        if self.dr is not None:
            self.dr.remove()
        
        # change graph based on change of position
        self.par, self.dr, self.move = self.drone.pos(self.img, event.key)
        self.img.figure.canvas.draw_idle()
        self.count = self.count + self.move
        if self.move == 0:    
            print("Out of bound, the drone keep its postion")
        else:
            print("The drone moved %d times" %  self.count) 

    def disconnect(self):
        self.img.figure.canvas.mpl_disconnect(self.cidpress)
########################################################################################################
#Drone class
class Drone:
    def __init__(self, imgdata, width, height, x ,y):
        self.imgdata = imgdata
        # width, height
        self.width = width
        self.height = height

        # initial x, y coordinates
        self.noise_range = 0 #np.random.uniform(-5,5) 
        self.x = x
        self.y = y 
        # observation image
        self.obs_img = self.imgdata[int(self.y - M//2 - self.noise_range):int(self.y + M//2+ self.noise_range), int(self.x - M//2- self.noise_range):int(self.x + M//2 + self.noise_range)]
        
        # call particle filter class
        self.pf = PF(self.imgdata, self.width, self.height)

    def pos(self, img, key):
        # calculate the weight of particle before update new position
        self.pf.likelihood(self.obs_img, self.x, self.y)
        # initalize dx dy
        dx = 0
        dy = 0
        if key == 'enter':
            dx, dy, move = self.drone_move()
        elif key == 'q':
            print("You exit the program \n")
        elif key == 's':
            print("You save \n")
        # resampling particles
        self.pf.resample()
        # move particle and update graph
        self.pf.particle_move(dx, dy)
        self.new_plot_par, self.new_plot_drone = self.plot(img)
        return self.new_plot_par, self.new_plot_drone, move
    
    def drone_move(self):
        # random factor
        dx = np.random.uniform(-1,1)
        dy = np.sqrt((1 - dx)**2)
        sign_random = np.random.uniform(0, 1)
        if sign_random >= 0.5:
            dy = dy
        else:
            dy = 0 - dy
        # keep old coordinate in case out of bound
        self.x_0 = self.x
        self.y_0 = self.y
        self.x = self.x + (dx + np.random.normal(0,0.4)) * 50
        self.y = self.y + (dy + np.random.normal(0,0.4)) * 50
        move = 1 #use to count how many time the drone was moved
        # if new coordinate out of bounds, ignore move
        if check_out_of_bound(self.x, self.width, self.y, self.height, M):
            self.x = self.x_0
            self.y = self.y_0
            dx = 0
            dy = 0
            move = 0
            return dx, dy, move
        return dx, dy, move

    def plot(self, img):
        # update image
        self.obs_img = self.imgdata[int(self.y - M//2):int(self.y + M//2), int(self.x - M//2):int(self.x + M//2)]
        # update graph
        show_par = self.pf.plot(img)
        show_drone = img.axes.scatter(self.x, self.y, s=200, color='black', marker='o')
        return show_par, show_drone

########################################################################################################
# Particle Filter class
class PF: 
    def __init__(self, imgdata, width, height):
        self.imgdata = imgdata
        # width, height
        self.width = width
        self.height = height
        self.particle_data = np.zeros((NUM_PARTICLE,2))
        self.particle_data[:,0] = np.random.uniform(M//2, self.width-M//2-1, size=NUM_PARTICLE)
        self.particle_data[:,1] = np.random.uniform(M//2, self.height-M//2-1, size=NUM_PARTICLE)
        #self.particle_data = np.array([np.random.uniform(M//2, self.width-M//2-1, size=NUM_PARTICLE), np.random.uniform(M//2, self.height-M//2-1, size=NUM_PARTICLE)]).T
        self.prob = np.empty(NUM_PARTICLE)
        self.weight = np.full_like(self.prob, NUM_PARTICLE*0.8)

    def likelihood(self, obs_img, x, y):
        for i, particle in enumerate(self.particle_data):
            ref_img = self.imgdata[int(particle[1]-M//2):int(particle[1]+M//2), int(particle[0]-M//2):int(particle[0]+M//2)]
            if self.width > 5000:
                ref_img_r = ref_img[:,:,0]
                ref_img_g = ref_img[:,:,1]
                ref_img_b = ref_img[:,:,2]
                obs_img_r = obs_img[:,:,0]
                obs_img_g = obs_img[:,:,1]
                obs_img_b = obs_img[:,:,2]
                rms_r = rmse(obs_img_r, ref_img_r)
                rms_g = rmse(obs_img_g, ref_img_g)
                rms_b = rmse(obs_img_b, ref_img_b)
                dominant_col = []
                dominant_col.extend([np.mean(obs_img_r, axis = (0,1)), np.mean(obs_img_g, axis = (0,1)) ,np.mean(obs_img_b, axis = (0,1))])
                if np.argsort(dominant_col[2]) % 3 == 0: #red dominant
                    rms = rms_r
                elif np.argsort(dominant_col[2]) % 3 == 1: # green dominant
                    rms = rms_g
                else: #blue dominant
                    rms = rms_b
            else:
                rms = rmse(obs_img, ref_img)
            self.prob[i] = 1 - np.max(rms)
        self.weight = 100*self.prob

        # find distance of the highest weigth particle and drone
        self.highest_weight = np.argsort(self.weight)
        self.highest_pos = self.particle_data[self.highest_weight[-1],:]
        self.distance = dist(x, self.highest_pos[0], y, self.highest_pos[1])
        print("Distance is %2.2f pixels" % self.distance)

    def resample(self):
        self.mix = np.random.uniform(0,1)
        if self.mix >= 0.05:
            j = SUS(self.weight, NUM_PARTICLE)
        else:
            j = RPF(self.particle_data, self.weight, self.width, self.height)
        self.weight = self.weight[j]
        self.particle_data = self.particle_data[j]
        
    def particle_move(self, dx, dy):
        for i, particle in enumerate(self.particle_data):
            # copy
            par_x0 = particle[0]
            par_y0 = particle[1]
            # update
            par_x = particle[0] + (dx + np.random.normal(0,0.2)) * 50
            par_y = particle[1] + (dy + np.random.normal(0,0.2)) * 50
            if check_out_of_bound(par_x, self.width, par_y, self.height, M):
                # don't move particle if out of bound
                #par_x = par_x0
                #par_y = par_y0
                continue
            #update
            self.particle_data[i] = np.array([par_x, par_y])

    def plot(self, img):
        show_part = img.axes.scatter(self.particle_data[:,0], self.particle_data[:,1], s=self.weight, color='r', marker = '*')
        return show_part

########################################################################################################
if __name__ == '__main__':
    width, height = get_pixels(image_name)
    im = plt.imread(image_name)
    run = Control(im, width, height)
    run.connect()
    plt.show()
