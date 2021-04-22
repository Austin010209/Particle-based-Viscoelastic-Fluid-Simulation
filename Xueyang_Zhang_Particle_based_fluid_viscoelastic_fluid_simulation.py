import os
import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

n = 500
dt = 0.01
Nparticle = 4520
grid_size = 50
cell_size = (int) (n/grid_size)
cell_size_recip = 1.0/cell_size
particle_size = 2.5
max_neighbor = 500
h = particle_size * 2
eps = 1e-10

Xmin_boundary = 10.0
Ymin_boundary = 10.0
Ymin_rigid_boundary = 10.0
Xmax_boundary = (n-Xmin_boundary)*1.0


mu = 0.0        #in collision
g = -100
sigma = 0.5   #in viscosity   #0.5
beta = 0.005    #in viscosity
gamma = 0.1    #in spring adjustment
alpha = 0.3    #in spring adjustment
k_spring = 30 #in spring displacement
rho0 = 1.34      #in double density relaxation
k = 200
k_near = 10
k_close = 1


frame_N = ti.field(dtype=ti.i32, shape=())
coef = ti.field(dtype=ti.i32, shape=())

oldpositions = ti.Vector.field(2, dtype=ti.f32, shape=Nparticle)
# extract particle position by its "ID"
positions = ti.Vector.field(2, dtype=ti.f32, shape=Nparticle)
# extract particle velocity by its "ID"
velocities = ti.Vector.field(2, dtype=ti.f32, shape=Nparticle)
# extract particle mass by its "ID"
Mass = ti.field(dtype=ti.f32, shape=Nparticle)
# extract # of particles in each cell of grid
grid_num_particles = ti.field(dtype=ti.i32, shape=(grid_size, grid_size))
# extract particle ID in one cell in grid by cell position, idx of neighbor
grid2particles = ti.field(dtype=ti.i32, shape=(grid_size, grid_size, max_neighbor))
# extract neighbors of a particle by its ID, idx of neighbor
particle_neighbors = ti.field(dtype=ti.i32, shape=(Nparticle, max_neighbor))
# extract # of neighbor of a particle by its ID
particle_num_neighbors = ti.field(dtype=ti.i32, shape=Nparticle)
# neighbor_pairs[pt1, pt2]==1 iff they are a neighbor pair
neighbor_pairs = ti.field(dtype=ti.i32, shape=(Nparticle, Nparticle))
# strings[pt1, pt2] == 0 iff there are no spring, else it is the length of rest length
springs = ti.field(dtype=ti.f32, shape=(Nparticle, Nparticle))


Nobj = ti.field(dtype=ti.i32, shape=1)
Nobj_reserve = 8
pinned = ti.field(dtype=ti.i32, shape=Nobj_reserve)
center = ti.Vector.field(2, dtype=ti.f32, shape=Nobj_reserve)
angle = ti.field(dtype=ti.f32, shape=Nobj_reserve)
oldcenter = ti.Vector.field(2, dtype=ti.f32, shape=Nobj_reserve)
oldangle = ti.field(dtype=ti.f32, shape=Nobj_reserve)
rotateM = ti.Matrix.field(2,2, dtype=ti.f32, shape=Nobj_reserve)
color = ti.Vector.field(3, dtype=ti.f32, shape=Nobj_reserve)
width = ti.field(dtype=ti.f32, shape=Nobj_reserve)
height = ti.field(dtype=ti.f32, shape=Nobj_reserve)
mass = ti.field(dtype=ti.f32, shape=Nobj_reserve)

bodyLB = ti.Vector.field(2, dtype=ti.f32, shape=Nobj_reserve)
bodyLU = ti.Vector.field(2, dtype=ti.f32, shape=Nobj_reserve)
bodyRU = ti.Vector.field(2, dtype=ti.f32, shape=Nobj_reserve)
bodyRB = ti.Vector.field(2, dtype=ti.f32, shape=Nobj_reserve)

LB = ti.Vector.field(2, dtype=ti.f32, shape=Nobj_reserve)
LU = ti.Vector.field(2, dtype=ti.f32, shape=Nobj_reserve)
RU = ti.Vector.field(2, dtype=ti.f32, shape=Nobj_reserve)
RB = ti.Vector.field(2, dtype=ti.f32, shape=Nobj_reserve)
side1 = ti.Vector.field(2, dtype=ti.f32, shape=Nobj_reserve)
side2 = ti.Vector.field(2, dtype=ti.f32, shape=Nobj_reserve)
side3 = ti.Vector.field(2, dtype=ti.f32, shape=Nobj_reserve)
side4 = ti.Vector.field(2, dtype=ti.f32, shape=Nobj_reserve)
velocity = ti.Vector.field(2, dtype=ti.f32, shape=Nobj_reserve)
omega = ti.field(dtype=ti.f32, shape=Nobj_reserve)
F = ti.Vector.field(2, dtype=ti.f32, shape=Nobj_reserve)
torque = ti.field(dtype=ti.f32, shape=Nobj_reserve)
I = ti.field(dtype=ti.f32, shape=Nobj_reserve)
boundary = ti.Vector.field(2, dtype=ti.f32, shape=(Nobj_reserve, 800))
cur_boundary = ti.Vector.field(2, dtype=ti.f32, shape=(Nobj_reserve, 800))
boundary_count = ti.field(dtype=ti.i32, shape=Nobj_reserve)

pts_inside = ti.field(dtype=ti.i32, shape=(Nobj_reserve,35,35,100))
pts_in_count = ti.field(dtype=ti.i32, shape=Nobj_reserve)

tmp_vector = ti.Vector.field(2, dtype=ti.f32, shape=4)

Count = ti.field(dtype=ti.i32, shape=(Nobj_reserve, 35, 35))
Count1 = ti.field(dtype=ti.i32, shape=Nobj_reserve)

@ti.func
def createboundary(rec_idx, a, b):
    for i in range(a+1):
        boundary[rec_idx, i] = ti.Vector([1.0*i, 0.0])
    for i in range(a+1):
        boundary[rec_idx, a+1+i] = ti.Vector([1.0*i, b*1.0])
    for i in range(1, b):
        boundary[rec_idx, 2*a+1 + i] = ti.Vector([0.0, 1.0*i])
    for i in range(1, b):
        boundary[rec_idx, 2*a+b + i] = ti.Vector([a*1.0, 1.0*i])
    for i in range(2*a + 2*b):
        boundary[rec_idx, i] += -ti.Vector([a, b])/2.0
    boundary_count[rec_idx] = 2*a + 2*b

@ti.func
def create_rectangle(rec_idx, ppinned, pcenter, pangle, pv, pomega, pcolor, pwidth, pheight, pmass=1000.0):
    pinned[rec_idx] = ppinned
    center[rec_idx] = pcenter
    oldcenter[rec_idx] = pcenter
    angle[rec_idx] = pangle
    oldangle[rec_idx] = pangle
    rotateM[rec_idx] = ti.Matrix([[ti.cos(pangle), -ti.sin(pangle)], [ti.sin(pangle), ti.cos(pangle)]])
    color[rec_idx] = pcolor
    width[rec_idx] = pwidth
    height[rec_idx] = pheight
    mass[rec_idx] = pmass
    bodyLB[rec_idx] = ti.Vector([-pwidth, -pheight]) /2
    bodyLU[rec_idx] = ti.Vector([-pwidth, pheight]) /2
    bodyRU[rec_idx] = ti.Vector([pwidth, pheight]) /2
    bodyRB[rec_idx] = ti.Vector([pwidth, -pheight]) /2
    
    LB[rec_idx] = pcenter + rotateM[rec_idx] @ bodyLB[rec_idx]
    LU[rec_idx] = pcenter + rotateM[rec_idx] @ bodyLU[rec_idx]
    RU[rec_idx] = pcenter + rotateM[rec_idx] @ bodyRU[rec_idx]
    RB[rec_idx] = pcenter + rotateM[rec_idx] @ bodyRB[rec_idx]
    side1[rec_idx] = (RU[rec_idx] - LU[rec_idx]).normalized()
    side2[rec_idx] = (RB[rec_idx] - RU[rec_idx]).normalized()
    side3[rec_idx] = (LB[rec_idx] - RB[rec_idx]).normalized()
    side4[rec_idx] = (LU[rec_idx] - LB[rec_idx]).normalized()
    velocity[rec_idx] = pv
    omega[rec_idx] = pomega
    F[rec_idx] = ti.Vector([0.0, 0.0])
    torque[rec_idx] = 0.0
    I[rec_idx] = (pwidth*pwidth + pheight*pheight) * pmass/12.0
    
    createboundary(rec_idx, pwidth, pheight)

@ti.func
def is_in_boundary(rec_idx, pt):
    vecLU = positions[pt] - LU[rec_idx]
    dist1 = abs(vecLU.dot(side1[rec_idx]))
    dist2 = abs(vecLU.dot(side2[rec_idx]))
    
    vecRB = positions[pt] - RB[rec_idx]
    dist3 = abs(vecRB.dot(side3[rec_idx]))
    dist4 = abs(vecRB.dot(side4[rec_idx]))
    
    fake_width = dist1 + dist3
    fake_height = dist2 + dist4
    Ans = False
    if fake_width < width[rec_idx] + 0.001 and fake_width > width[rec_idx] - 0.001 and fake_height < height[rec_idx] + 0.001 and fake_height > height[rec_idx] - 0.001:
        Ans = True
    return Ans

@ti.func
def prepare():
    for i, j, k in ti.ndrange(Nobj_reserve, 15, 15):
        Count[i, j, k] = 0
        
@ti.func
def particle_inside(rec_idx):
    prepare()
    pts_in_count[rec_idx] = 0
    
    maxx = max(LB[rec_idx][0], LU[rec_idx][0], RU[rec_idx][0], RB[rec_idx][0])
    minx = min(LB[rec_idx][0], LU[rec_idx][0], RU[rec_idx][0], RB[rec_idx][0])
    maxy = max(LB[rec_idx][1], LU[rec_idx][1], RU[rec_idx][1], RB[rec_idx][1])
    miny = min(LB[rec_idx][1], LU[rec_idx][1], RU[rec_idx][1], RB[rec_idx][1])
    
    Minx = int(ti.floor(minx))
    Miny = int(ti.floor(miny))
    Maxx = int(ti.ceil(maxx))
    Maxy = int(ti.ceil(maxy))
    theMin = ti.Vector([Minx, Miny])
    
    dif1 = ti.ceil((Maxx - Minx)*cell_size_recip)
    dif2 = ti.ceil((Maxy - Miny)*cell_size_recip)
    
    ct = 0
    base_cell = getcell(theMin)
    for offs in ti.grouped(ti.ndrange((0, dif1), (0, dif2))):
        cell_to_check = base_cell + offs
        if is_in_grid(cell_to_check):
            for j in range(grid_num_particles[cell_to_check]):
                p_j = grid2particles[cell_to_check, j]
                if is_in_boundary(rec_idx, p_j):
                    pts_inside[rec_idx, offs, Count[rec_idx, offs]] = p_j
                    Count[rec_idx, offs] += 1
                    ti.atomic_add(ct, 1)
    pts_in_count[rec_idx] = ct
    
   
@ti.func
def construct_rect(rec_idx):
    theta = angle[rec_idx]
    rotateM[rec_idx] = ti.Matrix([[ti.cos(theta), -ti.sin(theta)], [ti.sin(theta), ti.cos(theta)]])
    LB[rec_idx] = rotateM[rec_idx] @ bodyLB[rec_idx] + center[rec_idx]
    LU[rec_idx] = rotateM[rec_idx] @ bodyLU[rec_idx] + center[rec_idx]
    RU[rec_idx] = rotateM[rec_idx] @ bodyRU[rec_idx] + center[rec_idx]
    RB[rec_idx] = rotateM[rec_idx] @ bodyRB[rec_idx] + center[rec_idx]

    side1[rec_idx] = (RU[rec_idx] - LU[rec_idx]).normalized()
    side2[rec_idx] = (RB[rec_idx] - RU[rec_idx]).normalized()
    side3[rec_idx] = (LB[rec_idx] - RB[rec_idx]).normalized()
    side4[rec_idx] = (LU[rec_idx] - LB[rec_idx]).normalized()
    
    for i in range(boundary_count[rec_idx]):
        cur_boundary[rec_idx, i] = rotateM[rec_idx] @ boundary[rec_idx, i]
    for i in range(boundary_count[rec_idx]):
        cur_boundary[rec_idx, i] += center[rec_idx]

@ti.func
def advance_rect(rec_idx):
    if pinned[rec_idx]==0:
        center[rec_idx] += velocity[rec_idx]*dt
        angle[rec_idx] += omega[rec_idx]*dt
        construct_rect(rec_idx)
    
@ti.func
def rotate90(v):
    return ti.Vector([ -v[1], v[0] ])


@ti.func
def normal_direc(rec_idx, pt):
    vecLU = positions[pt] - LU[rec_idx]
    dist1 = abs(vecLU.dot(side1[rec_idx]))
    dist2 = abs(vecLU.dot(side2[rec_idx]))
    
    vecRB = positions[pt] - RB[rec_idx]
    dist3 = abs(vecRB.dot(side3[rec_idx]))
    dist4 = abs(vecRB.dot(side4[rec_idx]))
    direc = ti.Vector([0.0, 0.0])
    
    if(dist1 <= dist2 and dist1 <= dist3 and dist1 <= dist4):
        direc = side3[rec_idx]
    elif (dist2 <= dist1 and dist2 <= dist3 and dist2 <= dist4):
        direc = side4[rec_idx]
    elif (dist3 <= dist1 and dist3 <= dist2 and dist3 <= dist4):
        direc = side1[rec_idx]
    else:
        direc = side2[rec_idx]
    direc = direc.normalized()
    return direc

@ti.func
def first_phase_one_pt(rec_idx, pt):
    Ndir = normal_direc(rec_idx, pt)
    dist = positions[pt] - center[rec_idx]
    vp = velocity[rec_idx] + omega[rec_idx]*rotate90(dist)
    vbar = velocities[pt] - vp
    vbar_N = (vbar.dot(Ndir)) * Ndir
    vbar_T = vbar - vbar_N
    Impulse = vbar_N + mu*vbar_T
    
    F[rec_idx] += Impulse/dt
    torque[rec_idx] += (rotate90(dist)).dot(Impulse/dt)

@ti.func
def first_phase():
    for II in Count1:
        Count1[II] = 0
    
    for i, j, k, m in pts_inside:
        if pts_inside[i, j, k, m] != 0:
            Count1[i] += 1
            pt = pts_inside[i, j, k, m]
            first_phase_one_pt(i, pt)
    if Count1[0] != pts_in_count[0]:
        first_phase_one_pt(0, 0)

@ti.func
def second_phase_one_pt(rec_idx, pt):
    Ndir = normal_direc(rec_idx, pt)
    dist = positions[pt] - center[rec_idx]
    vp = velocity[rec_idx] + omega[rec_idx]*rotate90(dist)
    vbar = velocities[pt] - vp
    vbar_N = (vbar.dot(Ndir)) * Ndir
    vbar_T = vbar - vbar_N
    Impulse = vbar_N + mu*vbar_T
    Impulse *= -1
    velocities[pt] += Impulse
    if abs(Ndir[0]) > 0.97 and velocities[pt][1] > 10:
        velocities[pt][1] = 10
        if Ndir[0] > 0.97:
            velocities[pt][0] += 30
        else:
            velocities[pt][0] += -30
    positions[pt] += velocities[pt] * dt
    #if still in boundary
    while(is_in_boundary(rec_idx, pt)):
        positions[pt] += 0.1 * normal_direc(rec_idx, pt) * dt

@ti.func
def second_phase():
    #for each particle
    for I in Count1:
        Count1[I] = 0
    for i, j, k, m in pts_inside:
        if pts_inside[i, j, k, m] != 0:
            Count1[i] += 1
            pt = pts_inside[i, j, k, m]
            second_phase_one_pt(i, pt)


@ti.func
def bottom_body_collision(rec_idx, Ymin):
    for i in range(boundary_count[rec_idx]):
        if cur_boundary[rec_idx, i][1] < Ymin:
            ctct_obj1 = cur_boundary[rec_idx, i]
            ctct_obj2 = ti.Vector([ctct_obj1[0], Ymin])
            normal = ti.Vector([0.0, -1.0])
            dist = abs(ctct_obj1[1] - Ymin)
            ctctpt = ti.Vector([ctct_obj1[0], (ctct_obj1[1] + Ymin)/2])
            
            vtemp = (ctctpt - center[rec_idx]) * omega[rec_idx]
            relavel = rotate90(vtemp) + velocity[rec_idx]
            if -relavel.dot(normal) < 1.0e-9:
                # spring force
                force1 = (-dist*10*mass[0]) * normal
                F[rec_idx] += force1
                dif = ctctpt - center[rec_idx]
                delta_tao = -dif[1] * force1[0] + dif[0] * force1[1]
                torque[rec_idx] += delta_tao
                
                # spring damping forces!
                force1 = -(relavel.dot(normal)) * 10.0 * normal
                F[rec_idx] += force1
                dif = ctctpt - center[rec_idx]
                delta_tao = -dif[1] * force1[0] + dif[0] * force1[1]
                torque[rec_idx] += delta_tao
          
    velocity[rec_idx] += F[rec_idx]/mass[rec_idx]*dt
    velocity[rec_idx][0] *= 0.93
    omega[rec_idx] += torque[rec_idx]/I[rec_idx]*dt
      
@ti.func
def left_body_collision(rec_idx, Xmin):
    for i in range(boundary_count[rec_idx]):
        if cur_boundary[rec_idx, i][0] < Xmin:
            ctct_obj1 = cur_boundary[rec_idx, i]
            ctct_obj2 = ti.Vector([Xmin, ctct_obj1[1]])
            normal = ti.Vector([-1.0, 0.0])
            dist = abs(ctct_obj1[0] - Xmin)
            ctctpt = ti.Vector([(ctct_obj1[0] + Xmin)/2.0, ctct_obj1[1]])
            
            vtemp = (ctctpt - center[rec_idx]) * omega[rec_idx]
            relavel = rotate90(vtemp) + velocity[rec_idx]
            if -relavel.dot(normal) < 1.0e-9:
                # spring force
                force1 = (-dist*7*mass[0]) * normal
                F[rec_idx] += force1
                dif = ctctpt - center[rec_idx]
                delta_tao = -dif[1] * force1[0] + dif[0] * force1[1]
                torque[rec_idx] += delta_tao
                
                # spring damping forces!
                force1 = -(relavel.dot(normal)) * 10.0 * normal
                F[rec_idx] += force1
                dif = ctctpt - center[rec_idx]
                delta_tao = -dif[1] * force1[0] + dif[0] * force1[1]
                torque[rec_idx] += delta_tao      
    velocity[rec_idx] += F[rec_idx]/mass[rec_idx]*dt
    omega[rec_idx] += torque[rec_idx]/I[rec_idx]*dt
    
@ti.func
def right_body_collision(rec_idx, Xmax):
    for i in range(boundary_count[rec_idx]):
        if cur_boundary[rec_idx, i][0] > Xmax:
            ctct_obj1 = cur_boundary[rec_idx, i]
            ctct_obj2 = ti.Vector([Xmax, ctct_obj1[1]])
            normal = ti.Vector([1.0, 0.0])
            dist = abs(ctct_obj1[0] - Xmax)
            ctctpt = ti.Vector([(ctct_obj1[0] + Xmax)/2.0, ctct_obj1[1]])
            
            vtemp = (ctctpt - center[rec_idx]) * omega[rec_idx]
            relavel = rotate90(vtemp) + velocity[rec_idx]
            if -relavel.dot(normal) < 1.0e-9:
                # spring force
                force1 = (-dist*7*mass[0]) * normal
                F[rec_idx] += force1
                dif = ctctpt - center[rec_idx]
                delta_tao = -dif[1] * force1[0] + dif[0] * force1[1]
                torque[rec_idx] += delta_tao
                
                # spring damping forces!
                force1 = -(relavel.dot(normal)) * 10.0 * normal
                F[rec_idx] += force1
                dif = ctctpt - center[rec_idx]
                delta_tao = -dif[1] * force1[0] + dif[0] * force1[1]
                torque[rec_idx] += delta_tao      
    velocity[rec_idx] += F[rec_idx]/mass[rec_idx]*dt
    omega[rec_idx] += torque[rec_idx]/I[rec_idx]*dt
    

@ti.func
def resolve_body_collision(rec_idx):
    maxx = max(LB[rec_idx][0], LU[rec_idx][0], RU[rec_idx][0], RB[rec_idx][0])
    minx = min(LB[rec_idx][0], LU[rec_idx][0], RU[rec_idx][0], RB[rec_idx][0])
    miny = min(LB[rec_idx][1], LU[rec_idx][1], RU[rec_idx][1], RB[rec_idx][1])

    if miny < Ymin_boundary:
        bottom_body_collision(rec_idx, Ymin_boundary)
        center[rec_idx] += velocity[rec_idx] *dt
        angle[rec_idx] += omega[rec_idx] * dt
    if minx < Xmin_boundary:
        left_body_collision(rec_idx, Xmin_boundary)
        center[rec_idx] += velocity[rec_idx] *dt
        angle[rec_idx] += omega[rec_idx] * dt
    if maxx > Xmax_boundary:
        right_body_collision(rec_idx, Xmax_boundary)
        center[rec_idx] += velocity[rec_idx] *dt
        angle[rec_idx] += omega[rec_idx] * dt

@ti.func
def resolveCollisions():
    #clear buffer
    for II in ti.grouped(pts_inside):
        pts_inside[II] = 0
    for i in range(Nobj[0]):
        if pinned[i] == 0:
            # save original position and orientation
            oldcenter[i] = center[i]
            oldangle[i] = angle[i]
            # advance
            advance_rect(i)
            # clear F and torque buffer
            F[i] = [0.0, 0.0]
            torque[i] = 0.0
            particle_inside(i)
    first_phase()
    
   
    # second for each body
    for rec_idx in range(Nobj[0]):
        if pinned[rec_idx] == 0:
            velocity[rec_idx] += (F[rec_idx]/mass[rec_idx]) * dt
            omega[rec_idx] += torque[rec_idx]/I[rec_idx] * dt
            center[rec_idx] = oldcenter[rec_idx]
            angle[rec_idx] = oldangle[rec_idx]
            advance_rect(rec_idx)
    
    #it is unclear whether the position of body needs to be updated
    for rec_idx in range(Nobj[0]):
        if pinned[rec_idx] == 0:
            resolve_body_collision(rec_idx)
    
    for II in ti.grouped(pts_inside):
        pts_inside[II] = 0
    for rec_idx in range(Nobj[0]):
        particle_inside(rec_idx)
    second_phase()

@ti.func
def getcell(pos):
    return (int) (pos * cell_size_recip)

@ti.func
def is_in_grid(cell):
    return cell[0]>=0 and cell[0]<grid_size and cell[1]>=0 and cell[1]<grid_size

@ti.kernel
def create_system1():
    width = 40.0
    height = 1.0*Nparticle/width
    center = ti.Vector([250.0, 300.0])
    dif = ti.Vector([width, height]) /2.0 * particle_size
    LF = center - dif

    for i in range(Nparticle):
        positions[i] = (LF + ti.Vector([i//height, i % height]) * particle_size)
        velocities[i] = [0.0, 0.0]
       
    positions[0] = [10.0, 10.0]
    velocities[0] = [0.0, 0.0]
    
@ti.kernel
def create_system2():
    N = Nparticle*0.5
    width = 20.0
    height = 1.0*Nparticle*0.5/width
    center = ti.Vector([70.0, 300.0])
    dif = ti.Vector([width, height]) /2.0 * particle_size
    LF = center - dif
    for i in range(N):
        positions[i] = (LF + ti.Vector([i//height, i % height]) * particle_size)
        velocities[i] = [-10.0, 0.0]
    
    center = ti.Vector([430.0, 300.0])
    LF = center - dif
    for i in range(N):
        positions[i+N] = (LF + ti.Vector([i//height, i % height]) * particle_size)
        velocities[i+N] = [10.0, 0.0]
        
    positions[0] = [10.0, 10.0]
    velocities[0] = [0.0, 0.0]

@ti.kernel
def create_system3():
    width = 40.0
    height = 1.0*Nparticle/width
    center = ti.Vector([150.0, 300.0])
    dif = ti.Vector([width, height]) /2.0 * particle_size
    LF = center - dif
    for i in range(Nparticle):
        positions[i] = (LF + ti.Vector([i//height, i % height]) * particle_size)
        velocities[i] = [0.0, -50.0]
    
    positions[0] = [10.0, 10.0]
    velocities[0] = [0.0, 0.0]
    
    Nobj[0] = 1
    create_rectangle(0, 0, ti.Vector([350.0, 300.0]), 1.0, ti.Vector([-10.0, 50.0]), 1.0, ti.Vector([0.77, 0.17, 0.36]), 100.0, 60.0, 10000.0)
    
@ti.kernel
def create_system5():
    width = 40.0
    height = 1.0*Nparticle/width
    center = ti.Vector([150.0, 300.0])
    dif = ti.Vector([width, height]) /2.0 * particle_size
    LF = center - dif
    for i in range(Nparticle):
        positions[i] = (LF + ti.Vector([i//height, i % height]) * particle_size)
        velocities[i] = [0.0, -50.0]
    
    positions[0] = [10.0, 10.0]
    velocities[0] = [0.0, 0.0]
    
    Nobj[0] = 1
    create_rectangle(0, 0, ti.Vector([350.0, 300.0]), 1.0, ti.Vector([-10.0, 50.0]), 1.0, ti.Vector([0.77, 0.17, 0.36]), 100.0, 50.0, 100.0)
    
@ti.kernel
def create_system6():
    width = 40.0
    height = 1.0*Nparticle/width
    center = ti.Vector([250.0, 350.0])
    dif = ti.Vector([width, height]) /2.0 * particle_size
    LF = center - dif

    for i in range(Nparticle):
        positions[i] = (LF + ti.Vector([i//height, i % height]) * particle_size)
        velocities[i] = [50.0, 0.0]
    
    positions[0] = [10.0, 10.0]
    velocities[0] = [0.0, 0.0]
    
    Nobj[0] = 3
    create_rectangle(0, 0, ti.Vector([350.0, 250.0]), 1.0, ti.Vector([-10.0, 0.0]), 1.0, ti.Vector([0.77, 0.17, 0.36]), 100.0, 60.0, 10000.0)
    create_rectangle(1, 1, ti.Vector([105.0, 200.0]), 0.0, ti.Vector([0.0, 0.0]), 0.0, ti.Vector([0.84, 0.82, 0.34]), 190.0, 20.0, 100.0)
    create_rectangle(2, 1, ti.Vector([420.0, 350.0]), 0.0, ti.Vector([0.0, 0.0]), 0.0, ti.Vector([0.84, 0.82, 0.34]), 140.0, 20.0, 100.0)

@ti.kernel
def create_system7():
    width = 100.0
    height = 1.0*Nparticle/width
    center = ti.Vector([250.0, 500.0])
    dif = ti.Vector([width, height]) /2.0 * particle_size
    LF = center - dif

    for i in range(Nparticle):
        positions[i] = (LF + ti.Vector([i//height, i % height]) * particle_size)
        velocities[i] = [0.0, -10.0]
    
    positions[0] = [10.0, 10.0]
    velocities[0] = [0.0, 0.0]
    
    Nobj[0] = 4
    create_rectangle(0, 0, ti.Vector([300.0, 100.0]), 0.0, ti.Vector([0.0, 0.0]), 0.0, ti.Vector([0.77, 0.17, 0.36]), 50.0, 100.0, 300.0)
    create_rectangle(1, 1, ti.Vector([145.0, 390.0]), 3*3.14/4, ti.Vector([0.0, 0.0]), 0.0, ti.Vector([0.84, 0.82, 0.34]), 205.0, 20.0, 100.0)
    create_rectangle(2, 1, ti.Vector([355.0, 390.0]), 3.14/4, ti.Vector([0.0, 0.0]), 0.0, ti.Vector([0.84, 0.82, 0.34]), 205.0, 20.0, 100.0)
    create_rectangle(3, 1, ti.Vector([250.0, 240.0]), 0.0, ti.Vector([0.0, 0.0]), 0.0, ti.Vector([0.84, 0.82, 0.34]), 30.0, 50.0, 100.0)

@ti.kernel
def create_system8():
    width = 50.0
    height = 1.0*Nparticle/width
    center = ti.Vector([250.0, 540.0])
    dif = ti.Vector([width, height]) /2.0 * particle_size
    LF = center - dif

    for i in range(Nparticle):
        positions[i] = (LF + ti.Vector([i//height, i % height]) * particle_size)
        velocities[i] = [0.0, -20.0]
    
    positions[0] = [10.0, 10.0]
    velocities[0] = [0.0, 0.0]
    
    Nobj[0] = 7
    create_rectangle(0, 0, ti.Vector([250.0, 70.0]), 0.0, ti.Vector([0.0, 0.0]), 0.0, ti.Vector([0.77, 0.17, 0.36]), 100.0, 52.0, 10000.0)
    create_rectangle(1, 1, ti.Vector([250.0, 420.0]), 0.0, ti.Vector([0.0, 0.0]), 0.0, ti.Vector([0.84, 0.82, 0.34]), 300.0, 40.0, 100.0)
    create_rectangle(2, 1, ti.Vector([104.0, 340.0]), 0.0, ti.Vector([0.0, 0.0]), 0.0, ti.Vector([0.84, 0.82, 0.34]), 200.0, 40.0, 100.0)
    create_rectangle(3, 1, ti.Vector([396.0, 340.0]), 0.0, ti.Vector([0.0, 0.0]), 0.0, ti.Vector([0.84, 0.82, 0.34]), 200.0, 40.0, 100.0)
    create_rectangle(4, 1, ti.Vector([250.0, 250.0]), 0.0, ti.Vector([0.0, 0.0]), 0.0, ti.Vector([0.84, 0.82, 0.34]), 260.0, 35.0, 100.0)
    create_rectangle(5, 1, ti.Vector([91.0, 150.0]), 0.0, ti.Vector([0.0, 0.0]), 0.0, ti.Vector([0.84, 0.82, 0.34]), 180.0, 35.0, 100.0)
    create_rectangle(6, 1, ti.Vector([409.0, 150.0]), 0.0, ti.Vector([0.0, 0.0]), 0.0, ti.Vector([0.84, 0.82, 0.34]), 180.0, 35.0, 100.0)
    
@ti.func
def prologue():
    for I in ti.grouped(springs):
        springs[I] = 0
    for I in ti.grouped(neighbor_pairs):
        neighbor_pairs[I] = 0
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1
    for I in ti.grouped(particle_num_neighbors):
        particle_num_neighbors[I] = 0
    #the clearance of other fields are not necessary, but it is only when you do all others correct...
    
    for i in positions:
        cell = getcell(positions[i])
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = i
    
    for p_i in positions:
        pos = positions[p_i]
        cell = getcell(pos)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):  #maybe need to check other boundary situation as well
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_neighbor and p_j != p_i and (
                            pos - positions[p_j]).norm(eps) < h:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
                        
                        neighbor_pairs[p_i, p_j] = 1
                        neighbor_pairs[p_j, p_i] = 1
                        
        particle_num_neighbors[p_i] = nb_i

@ti.func
def addforce():
    for i in range(Nobj[0]):
        if pinned[i] == 0:
            velocity[i][1] += dt*g
    for i in range(Nparticle):
        velocities[i][1] += dt*g


@ti.func
def applyViscosity():       
    for pt1, pt2 in neighbor_pairs:
        if pt1 < pt2 and neighbor_pairs[pt1, pt2]==1:
            rij = positions[pt2] - positions[pt1]
            q = rij.norm(eps)/h
            if q<1:
                vdif = velocities[pt1] - velocities[pt2]
                rdirection = rij / rij.norm(eps)
                u = rdirection[0]*vdif[0] + rdirection[1]*vdif[1]
                if u>0:
                    I = dt*(1-q)*(sigma+beta*u)*u*rdirection
                    velocities[pt1] += -I/2
                    velocities[pt2] += I/2

@ti.func
def advance():
    for i in positions:
        oldpositions[i] = positions[i]
        positions[i] += dt*velocities[i]
    
@ti.func
def adjustSprings(): #add and remove springs, change rest lengths
    for pt1, pt2 in neighbor_pairs:
        if pt1 < pt2 and neighbor_pairs[pt1, pt2]==1:
            rij = (positions[pt2] - positions[pt1]).norm(eps)
            q = rij/h
            if q < 1:   
                if springs[pt1, pt2]==0:
                    springs[pt1, pt2] = h
                L = springs[pt1, pt2]
                #if L<0.8*particle_size:
                # tolerable deformation = yield ratio * rest length
                d = gamma * L
                if rij > L + d:
                    springs[pt1, pt2] += dt*alpha*(rij - L - d)
                elif rij < L - d:
                    springs[pt1, pt2] += -dt*alpha*(L - d - rij)
    
    for pt1, pt2 in springs:
        if pt1 < pt2 and springs[pt1, pt2]!=0 and springs[pt1, pt2]>h:
            springs[pt1, pt2] = 0

@ti.func
def applySpringDisplacements():
    for pt1, pt2 in springs:
        if pt1 < pt2 and springs[pt1, pt2]!=0:
            Lij = springs[pt1, pt2]
            dif = positions[pt2] - positions[pt1]
            rij = dif.norm(eps)
            direction = dif/rij 
            D = dt*dt*k_spring* (1-Lij/h) * (Lij-rij) * direction
            positions[pt1] += -D/2
            positions[pt2] += D/2
    
@ti.func
def doubleDensityRelaxation():
    for i in positions:
        rho = 0.0
        rho_near = 0.0
        rho_close = 0.0
        # compute density and near-density
        pt1 = i
        for j in range(particle_num_neighbors[i]):
            pt2 = particle_neighbors[i, j]
            rij = (positions[pt2] - positions[pt1]).norm(eps)
            q = rij/h
            Q = 1-q
            if q<1:
                rho += Q*Q
                rho_near += Q*Q*Q
                if q<0.5:                     #
                    rho_close += Q*Q        #
                
        # compute pressure and near-pressure
        P = k * (rho-rho0)
        P_near = k_near * rho_near
        P_close = k_close * rho_close * coef[None]
        dx = ti.Vector([0.0, 0.0])
        D = ti.Vector([0.0, 0.0])
        
        for j in range(particle_num_neighbors[i]):
            pt2 = particle_neighbors[i, j]
            dif = positions[pt2] - positions[pt1]
            rij = dif.norm(eps)
            direction = dif/rij
            q = rij/h
            Q = 1-q
            if q<1:
                # apply displacements
                D = dt * dt * (P*Q + P_near*Q*Q + P_close*Q) * direction  #
                positions[pt2] += D/2
                dx += -D/2
        positions[pt1] += dx
          

@ti.func
def update_velocity():
    for i in positions:
        velocities[i] = (positions[i] - oldpositions[i]) / dt

@ti.func
def check_particle_boundary():
    for i in range(Nparticle):
        if (positions[i][1] < Ymin_boundary):
            positions[i][1] = Ymin_boundary + (Ymin_boundary - positions[i][1])
            velocities[i][1] *= -0.5
        if (positions[i][0] < Xmin_boundary):
            positions[i][0] = Xmin_boundary + (Xmin_boundary - positions[i][0])
            velocities[i][0] *= -0.5
        elif (positions[i][0] > Xmax_boundary):
            positions[i][0] = Xmax_boundary + (Xmax_boundary - positions[i][0])
            velocities[i][0] *= -0.5

@ti.kernel
def substep():
    prologue()
    #atest()
    addforce()                  # update v according to g
    applyViscosity()            # update v according to viscosity
    advance()                   # move particles
    
    prologue()
    adjustSprings()            # modify spring rest lengths
    applySpringDisplacements()  # apply spring forces(in position)
    
    prologue()
    doubleDensityRelaxation()   # Volume conservation, anti-clustering, and surface tension
    # collision between bodies and particles
    
    update_velocity()           # recompute velocity(for particles only)
    
    resolveCollisions()
    check_particle_boundary()
    
    if(frame_N[None] > 200 and frame_N[None] < 800):
        coef[None] += 1
    

def run_one_step():
    substep()

def draw_boundary(gui):
    start = np.array([[6.0, 500.0], [6.0, 6.0], [494.0, 6.0]])/n
    end = np.array([[6.0, 8.0], [494.0, 6.0], [494.0, 500.0]])/n
    gui.lines(start, end, color = 0x000000, radius = 4)

def draw_rectangles(gui):
    a = LB.to_numpy()/n
    b = LU.to_numpy()/n
    c = RU.to_numpy()/n
    d = RB.to_numpy()/n
    hex1 = ti.rgb_to_hex(np.transpose(color.to_numpy()))    
    gui.triangles(a, b, c, color = hex1)
    gui.triangles(c, d, a, color = hex1)
    gui.lines(a, c, color = hex1, radius = 1)

def draw_particles(gui):
    pos_np = positions.to_numpy()/n
    gui.circles(pos_np, radius = particle_size, color = 0X2FA2E5)

def render(gui):
    draw_particles(gui)
    draw_rectangles(gui)
    draw_boundary(gui)
    #gui.show()

def main():
    iterations = 20000
    create_system8()
    gui = ti.GUI('Particle-based Viscoelastic Fluid Simulation', (n, n), background_color = 0XC4FED4)
    frame_N[None] = 0
    coef[None] = 1
    for i in range(iterations):
        if gui.running:
            run_one_step()
            frame_N[None] += 1
            render(gui)
            # above if for generating video
            # filename = f'pictures/new_sytem7/frame_{i:05d}.png'
            # print(f'Frame {i} is recorded in {filename}')
            # gui.show(filename)
            gui.show()
        
        
        


if __name__ == '__main__':
    main()
    