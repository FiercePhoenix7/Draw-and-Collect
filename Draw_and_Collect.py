import pygame
import pymunk
import cv2
import mediapipe as mp
from rdp import rdp
import random
import math

####################### SETTINGS ###########################
pygame.init()
WIDTH, HEIGHT = 800, 600

DARK_GREEN = (0, 153, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

pygame.display.set_caption("Draw and Collect")

FPS = 60

################### HLEPER FUNCTIONS #######################

def dist(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

def create_ground(space):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Segment(body, (0, HEIGHT - 50), (WIDTH, HEIGHT - 50), 5)
    shape.elasticity = 0.9
    shape.friction = 0.8
    space.add(body, shape)
    return shape

def create_cup(base_rad, rim_rad, height, bcp):
    points = [(bcp[0]-rim_rad, bcp[1] - height), (bcp[0]-base_rad, bcp[1]), (bcp[0]+base_rad, bcp[1]), (bcp[0] + rim_rad, bcp[1] - height)]
    return create_curve(points)

def create_ball(pos, radius):
    mass = 0.06*(radius**2)
    body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, radius))
    body.position = pos
    shape = pymunk.Circle(body, radius)
    shape.elasticity = 0.5
    shape.friction = 0.8

    return (body, shape)

def create_curve(points):
    unit_mass = 0.1
    segment_radius = 3.0
    total_mass = 0
    total_moment = 0
    total_mass_times_position = pymunk.Vec2d.zero()
    segment_data = []

    for i in range(len(points) - 1):
        p1 = pymunk.Vec2d(*points[i])
        p2 = pymunk.Vec2d(*points[i+1])

        mass = unit_mass*(p1.get_distance(p2))
        
        segment_centre = (p1 + p2)/2.0

        segment_data.append((mass, p1, p2, segment_centre))
    
        total_mass += mass
        total_mass_times_position += segment_centre * mass
    
    centroid = total_mass_times_position / total_mass
    for data in segment_data:
        mass = data[0]
        p1 = data[1]
        p2 = data[2]
        segment_centre = data[3]

        p1_local = p1 - segment_centre
        p2_local = p2 - segment_centre
        moment_local = pymunk.moment_for_segment(mass, p1_local, p2_local, segment_radius)
        r = segment_centre - centroid
        moment_transfer = mass * r.dot(r)
        total_moment += moment_local + moment_transfer

    curve_body = pymunk.Body(total_mass, total_moment)
    curve_body.position = centroid
    segments = []
    for i in range(len(points) - 1):
        p1_local = pymunk.Vec2d(*points[i]) - curve_body.position
        p2_local = pymunk.Vec2d(*points[i+1]) - curve_body.position

        segment = pymunk.Segment(curve_body, p1_local, p2_local, segment_radius)
        segment.friction = 0.8
        segment.elasticity = 0.5
        segment.mass = segment_data[i][0]
        segments.append(segment)
    
    return (curve_body, segments)

def cross(a, b, c):
    ab = b - a
    ac = c - a
    return ab.cross(ac)

def is_inside_quad(quad, point):

    signs = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        signs.append(cross(a, b, point) >= 0)

    return all(signs) or not any(signs)

###################### APP ###########################

class App:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands #type: ignore
        self.Hands = self.mp_hands.Hands()
        self.fingertip = None
        self.pressed = False
        self.running = True
        self.Start_BG = pygame.image.load("Start_BG.png")
        self.Start_BG = pygame.transform.scale(self.Start_BG, (WIDTH, HEIGHT))
        self.Play_BG = pygame.image.load("Play_BG.png")
        self.Play_BG = pygame.transform.scale(self.Play_BG, (WIDTH, HEIGHT))
        self.level = 1
        self.playing = False
        self.score = 0

    def run(self):
        dt = 1.0 / FPS
        while self.running:
            success, self.frame = self.cap.read()
            if not success:
                continue
            self.frame = cv2.flip(self.frame, 1)
            self.non_playing_handle()
            while self.playing:
                success, self.frame = self.cap.read()
                if not success:
                    continue
                self.frame = cv2.flip(self.frame, 1)
                self.events()
                self.update()
                self.draw()
                self.space.step(dt)
                self.clock.tick(FPS)
            self.clock.tick(FPS)
        pygame.quit()
    
    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.playing = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.pressed = True
                elif event.key == pygame.K_x:
                    self.playing = False
                    self.score += self.get_score()
            elif event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                self.pressed = False
                if self.curve_points != []:
                    self.space.remove(*self.current_stroke_segments)
                    self.current_stroke_segments = []
                    simplified_points = rdp(self.curve_points, epsilon=4.0) # type: ignore
                    if len(simplified_points) > 1:
                        curve_body, segments = create_curve(simplified_points)
                        self.space.add(curve_body, *segments)
                        self.drawn_dynamic_bodies.append((curve_body, segments))
                        self.curve_points = []
                    else:
                        body, shape = create_ball(simplified_points[0], 5)
                        self.space.add(body, shape)
                        self.drawn_dynamic_bodies.append((body, shape))
                        self.curve_points = []


    def update(self):
        results = self.Hands.process(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        self.fingertip = None
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            self.fingertip = (int(lm[8].x*WIDTH), int(lm[8].y*WIDTH))
        
            if self.pressed and (self.fingertip not in self.curve_points):
                if self.curve_points == []: 
                    self.curve_points.append(self.fingertip) # type: ignore
                elif dist(self.curve_points[-1], self.fingertip) > 10:
                    self.curve_points.append(self.fingertip) # type: ignore
                    segment = pymunk.Segment(self.space.static_body, self.curve_points[-2], self.curve_points[-1], 2)
                    segment.friction = 0.8
                    segment.elasticity = 0.5
                    self.space.add(segment)
                    self.current_stroke_segments.append(segment)

    def draw(self):
        self.screen.fill((217, 217, 217))
        self.screen.blit(self.Play_BG, (0, 0))
        if self.fingertip:
            pygame.draw.circle(self.screen, RED, self.fingertip, 10, 5)
        
        if len(self.curve_points) > 1:
            pygame.draw.lines(
                self.screen,
                DARK_GREEN,
                False,
                self.curve_points, # type: ignore
                2
            )

        for body, shape in self.drawn_dynamic_bodies:
            if isinstance(shape, pymunk.Circle):
                pygame.draw.circle(self.screen, BLACK, (body.position.x, body.position.y), int(shape.radius))
            else:
                for seg in shape:
                    p1 = body.position + seg.a.rotated(body.angle)
                    p2 = body.position + seg.b.rotated(body.angle)
                    pygame.draw.line(self.screen, BLACK, p1, p2, int(seg.radius * 2))
        
        for body, shape in self.game_objects:
            if isinstance(shape, pymunk.Circle):
                pygame.draw.circle(self.screen, BLUE, (body.position.x, body.position.y), int(shape.radius))
            else:
                for seg in shape:
                    p1 = body.position + seg.a.rotated(body.angle)
                    p2 = body.position + seg.b.rotated(body.angle)
                    pygame.draw.line(self.screen, BLUE, p1, p2, int(seg.radius * 2))

        for shape in self.fixed_shapes:
            if isinstance(shape, pymunk.Segment):
                p1 = shape.body.position + shape.a.rotated(shape.body.angle) # type: ignore
                p2 = shape.body.position + shape.b.rotated(shape.body.angle) # type: ignore
                pygame.draw.line(self.screen, BLACK, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), int(shape.radius * 2))

        self.draw_text(f"Balls in Cup : {self.get_score()}", self.screen, (WIDTH - 175, 20), 25, RED, "times new roman")
        self.draw_text(f"Score : {self.score}", self.screen, (18, 20), 25, RED, "times new roman")

        pygame.display.flip()

    def new_level(self):
        if self.level == 1:
            self.curve_points = []
            self.drawn_curves = []
            self.drawn_dynamic_bodies = []
            self.game_objects = []
            body, segments = create_cup(35, 60, 80, (random.randint(100, WIDTH - 100), HEIGHT - 57))
            self.space.add(body, *segments)
            self.game_objects.append((body, segments))
            body, shape = create_ball((random.randint(50, WIDTH - 50), 0), 12)
            self.space.add(body, shape)
            self.game_objects.append((body, shape))
            self.current_stroke_segments = []
            ground_shape = create_ground(self.space)
            self.fixed_shapes = [ground_shape]
        else:
            print(self.level)
            for body, shape in self.drawn_dynamic_bodies:
                if isinstance(shape, pymunk.Circle):
                    self.space.remove(body, shape)
                else:
                    self.space.remove(body, *shape)
            for body, shape in self.game_objects:
                if isinstance(shape, pymunk.Circle):
                    self.space.remove(body, shape)
                else:
                    self.space.remove(body, *shape)
            if self.current_stroke_segments != []:
                self.space.remove(*self.current_stroke_segments)
            self.curve_points = []
            self.drawn_curves = []
            self.drawn_dynamic_bodies = []
            self.game_objects = []
            body, segments = create_cup(35, 60, 80, (random.randint(100, WIDTH - 100), HEIGHT - 57))
            self.space.add(body, *segments)
            self.game_objects.append((body, segments))
            for i in range(self.level):
                body, shape = create_ball((random.randint(50, WIDTH - 50), 0), 12)
                self.space.add(body, shape)
                self.game_objects.append((body, shape))
            self.current_stroke_segments = []
    
    def non_playing_handle(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_x:
                    self.new_level()
                    self.level += 1
                    self.playing = True 
        
        self.fingertip = None

        results = self.Hands.process(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            self.fingertip = (int(lm[8].x*WIDTH), int(lm[8].y*WIDTH))

        self.screen.fill((217, 217, 217))
        self.screen.blit(self.Start_BG, (0, 0))

        if self.fingertip:
            pygame.draw.circle(self.screen, RED, self.fingertip, 10, 5)

        pygame.display.flip()

    def get_score(self):
        cup, segments = self.game_objects[0]
        vertices = []
        for seg in segments:
            p1 = cup.position + seg.a.rotated(cup.angle)
            vertices.append(p1)
        p2 = cup.position + segments[-1].b.rotated(cup.angle)
        vertices.append(p2)

        score = 0
        for ball, shape in self.game_objects[1:]:
            if is_inside_quad(vertices, ball.position):
                score += 1
        
        return score

            
    def draw_text(self,words,screen, pos,size,colour,font_name):
        font = pygame.font.SysFont(font_name,size)
        text = font.render(words,0,colour)
        screen.blit(text,pos)

app = App()
app.run()

