import Particle
import sys


def PSO(num_particles, k, w, c1, c2, max_iter, convergence_criteria, data):
    swarm = []
    global_best = []
    global_fitness = sys.maxsize

    # Create particle swarm and initialize global best position
    for i in range(num_particles):
        cur_particle = Particle.Particle(k, w, c1, c2, data)
        swarm.append(cur_particle)

        # print(cur_particle.position)
        # print(type(cur_particle))
        particle_fitness = cur_particle.fitness(data, cur_particle.position)
        if particle_fitness < global_fitness:
            global_best = cur_particle.position
            global_fitness = particle_fitness

    # Run particle swarm
    for i in range(max_iter):
        for particle in swarm:
            particle.update_particle(global_best, data)
            if particle.fitness_val < global_fitness:
                global_best = particle.position
                global_fitness = particle.fitness

    return global_best






