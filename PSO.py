import Particle
import sys


def PSO(num_particles, k, w, c1, c2, max_iter, convergence_criteria, data):
    swarm = []
    global_best = []
    global_fitness = sys.maxsize
    g_fit = []

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

        g_fit.append(global_fitness)

    print("init global best: %s, fitness: %s" % (str(global_best), str(global_fitness)))

    # Run particle swarm
    for i in range(max_iter):
        print("iteration: %s" % i)
        # print("global best: %s, fitness: %s\n" % (str(global_best), global_fitness))
        for particle in swarm:
            if particle.fitness_val < global_fitness:
                # print("Particle position: %s, Particle fitness: %s" % (str(particle.position), str(particle.fitness_val)))
                global_best = particle.position
                global_fitness = particle.fitness_val
        g_fit.append(global_fitness)

        for particle in swarm:
            particle.update_particle(global_best, data)

    return global_best, g_fit








