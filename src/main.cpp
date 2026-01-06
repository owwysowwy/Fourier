#include <iostream>
#include <fstream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <shader/shader.h>
#include <fftw/fftw3.h>
#define NANOSVG_IMPLEMENTATION
#include <nanosvg/nanosvg.h>
#include <cmath>
#include <stdio.h>
#include <random>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


using namespace std; 

const char* WINDOW_NAME = "OPENGL WINDOW";
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 800;
const float twoPi = 2.0f * static_cast<float>(M_PI);
//seconds to complete one radian
const float timePerCycle = 3;
const int fps = 60;

bool recording = true;

const int NUM_CIRCLES = 1024;

const char *vertexCodeString = "./shaders/shader.vert";
const char *fragmentCodeString = "./shaders/shader.frag";

const char *pathVertCodeString = "./shaders/path.vert";
const char *pathFragCodeString = "./shaders/path.frag";

// Load SVG
NSVGimage *image = nsvgParseFromFile("./src/SVGS/7_qatar.svg", "px", 192);
const int countryNum = 7;

struct Point {
    float x;
    float y;
};

Point cubicBezier(
    const Point& p0,
    const Point& p1,
    const Point& p2,
    const Point& p3,
    float t
){
    float u = 1.0f - t;
    float uu = u * u;
    float uuu = uu * u;
    float tt = t * t;
    float ttt = tt * t;

    Point p;
    p.x = uuu * p0.x
        + 3 * uu * t * p1.x
        + 3 * u * tt * p2.x
        + ttt * p3.x;

    p.y = uuu * p0.y
        + 3 * uu * t * p1.y
        + 3 * u * tt * p2.y
        + ttt * p3.y;

    return p;
}

vector<Point> processSVG(NSVGimage* image){
    std::vector<Point> points;

    int samplesPerCurve = 4096;

    NSVGshape *shape = image->shapes;
    NSVGpath *path = shape->paths;
    for (int x = 0; x < countryNum; x++){
        path = path->next;
    }
    for (int i = 0; i < path->npts-1; i += 3) {
        float* p = &path->pts[i*2];

        Point p0 = {p[0], p[1]};
        Point p1 = {p[2], p[3]};
        Point p2 = {p[4], p[5]};
        Point p3 = {p[6], p[7]};

        for (int j = 0; j < samplesPerCurve; j++){
            points.push_back(cubicBezier(p0, p1, p2, p3, (float)j/samplesPerCurve));
        }


    }
    return points;
}

float dist2(const Point& a, const Point& b){
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx*dx + dy*dy;
}

vector<Point> removeDuplicates(const vector<Point>& points, float eps = 1e-6f){
    vector<Point> out;
    if (points.empty()) return out;

    out.push_back(points[0]);

    for (size_t i = 1; i < points.size(); i++){
        if (dist2(points[i], out.back()) > eps){
            out.push_back(points[i]);
        }
    }
    return out;
}

Point find_midpoint(vector<Point> points){
    float x_val = 0;
    float y_val = 0;
    for (int x = 0; x < points.size(); x++){
        x_val += points[x].x;
        y_val += points[x].y;
    }
    x_val /= points.size();
    y_val /= points.size();
    Point p;
    p.x = x_val;
    p.y =y_val;
    return p;
}

vector<Point> normalize_points(vector<Point> points){
    float max_dist = 0;
    Point midpoint = find_midpoint(points);
    for (int x = 0; x < points.size(); x++){
        points[x].x -= midpoint.x;
        points[x].y -= midpoint.y;
        float temp = sqrt(pow(points[x].x, 2) + pow(points[x].y, 2));
        max_dist = max(max_dist, temp);
    }
    max_dist *= 1.2;
    vector<Point> new_points;
    for (int x = 0; x < points.size(); x++){
        Point p;
        p.x = points[x].x / max_dist;
        p.y = points[x].y / max_dist;
        new_points.push_back(p);
    }
    return new_points;
}

vector<float> computeArcLengths(const vector<Point>& points){
    vector<float> lengths(points.size());
    lengths[0] = 0.0f;

    for (size_t i = 1; i < points.size(); i++){
        float dx = points[i].x - points[i-1].x;
        float dy = points[i].y - points[i-1].y;
        lengths[i] = lengths[i-1] + std::sqrt(dx*dx + dy*dy);
    }
    return lengths;
}

vector<Point> resampleByArcLength(const vector<Point>& points){
    vector<Point> out;
    if (points.size() < 2 || NUM_CIRCLES < 2)
        return out;

    vector<float> arc = computeArcLengths(points);
    float totalLength = arc.back();

    out.reserve(NUM_CIRCLES);

    float step = totalLength / (NUM_CIRCLES - 1);
    float currentDist = 0.0f;
    size_t seg = 0;

    for (int i = 0; i < NUM_CIRCLES; i++){
        while (seg + 1 < arc.size() && arc[seg + 1] < currentDist)
            seg++;

        if (seg + 1 >= arc.size()){
            out.push_back(points.back());
            break;
        }

        float t = (currentDist - arc[seg]) /
                (arc[seg + 1] - arc[seg]);

        Point p;
        p.x = points[seg].x +
              t * (points[seg + 1].x - points[seg].x);
        p.y = points[seg].y +
              t * (points[seg + 1].y - points[seg].y);

        out.push_back(p);
        currentDist += step;
    }

    return out;
}


float find_angle(fftw_complex arr){
    return std::atan2(arr[1], arr[0]);
}

fftw_complex *fft_test(int N, vector<Point> points){
    fftw_complex *in, *out;
    fftw_plan p;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int x = 0; x < N; x++){
        in[x][0] = points[x].x;
        in[x][1] = -points[x].y;
    }

    fftw_execute(p);

    fftw_destroy_plan(p);
    fftw_free(in); 
    return out;
}

int mapIndex(int i, int N){
    int fftIndex;
    if (i == 0)
        fftIndex = 0;
    else if (i % 2 == 1)
        fftIndex = (i + 1) / 2;     // positive
    else
        fftIndex = N - (i / 2);        // negative
    return fftIndex;
}

class Circle
{
public:
    int ID;
    float starting_angle;
    glm::vec3 position;
    float radius;
    float frequency;

    Circle(int ID, fftw_complex *output, int N) 
    {
        int k = mapIndex(ID, N);
        double *complex = output[k];

        float x = 0;
        float y = 0;
        // float normal = sqrt(pow(float(output[1][0]), 2) + pow(float(output[1][1]), 2)) * 2;
        int prevI;
        for (int i = ID - 1; i >= 0; i--){

            prevI = mapIndex(i, N);

            x += output[prevI][0];
            y += output[prevI][1];
        }
        position = glm::vec3(0.0f);

        this->ID = k;
        this->starting_angle = find_angle(complex);
        this->radius = sqrt(pow(complex[0], 2) + pow(complex[1], 2)) / NUM_CIRCLES;
        float freq;
        if (k <= N / 2)
            freq = k;
        else
            freq = k - N;
        this->frequency = freq / timePerCycle;
    }
};

// -------------------- CircleRenderer (ALL FIXED) --------------------
class CircleRenderer
{
public:
    CircleRenderer() = default;

    ~CircleRenderer()
    {
        if (VAO)         glDeleteVertexArrays(1, &VAO);
        if (meshVBO)     glDeleteBuffers(1, &meshVBO);
        if (instanceVBO) glDeleteBuffers(1, &instanceVBO);
        if (myShader.ID) glDeleteProgram(myShader.ID);
    }

    void init(int w, int h, const char* vsFile, const char* fsFile)
    {
        windowWidth  = w;
        windowHeight = h;

        myShader.init(vsFile, fsFile);
        setupCircleMesh();
        setupInstanceBuffer();
        updateProjection(windowWidth, windowHeight);
    }

    void setCircles(const std::vector<Circle>& c)
    {
        circles = c;
        updateInstanceBuffer();
    }

    // Call this if the window is resized
    void onResize(int w, int h)
    {
        windowWidth  = (w > 0) ? w : 1;
        windowHeight = (h > 0) ? h : 1;
        updateProjection(windowWidth, windowHeight);
    }

    Shader& getShader() {
        return myShader; 
    }

    void draw(float time)
    {
        if (!myShader.ID || !VAO) return;

        myShader.use();

        int timeLoc = glGetUniformLocation(myShader.ID, "time");
        glUniform1f(timeLoc, time);

        glBindVertexArray(VAO);
        glDrawArraysInstanced(
            GL_LINE_STRIP,
            0,
            vertexCount,
            static_cast<GLsizei>(circles.size())
        );
    }
    
private:
    void setupCircleMesh()
    {
        std::vector<float> verts;
        verts.reserve((SEGMENT_NUMBER + 2) * 3);

        verts.push_back(0.0f);
        verts.push_back(0.0f);
        verts.push_back(0.0f);

        for (int i = 0; i <= SEGMENT_NUMBER; i++)
        {
            float a = twoPi * i / SEGMENT_NUMBER;
            verts.push_back(std::cos(a));
            verts.push_back(std::sin(a));
            verts.push_back(0.0f);
        }

        vertexCount = static_cast<int>(verts.size() / 3);

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &meshVBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, meshVBO);
        glBufferData(GL_ARRAY_BUFFER,
                     verts.size() * sizeof(float),
                     verts.data(),
                     GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                              3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
    }

    void setupInstanceBuffer()
    {
        glGenBuffers(1, &instanceVBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);

        // allocate 0 for now; real size in updateInstanceBuffer()
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

        // Attribute 1: Circle.position (vec3)
        glVertexAttribPointer(
            1, 3, GL_FLOAT, GL_FALSE,
            sizeof(Circle), (void*)offsetof(Circle, position)
        );
        glEnableVertexAttribArray(1);
        glVertexAttribDivisor(1, 1);

        // Attribute 2: Circle.radius (float)
        glVertexAttribPointer(
            2, 1, GL_FLOAT, GL_FALSE,
            sizeof(Circle), (void*)offsetof(Circle, radius)
        );
        glEnableVertexAttribArray(2);
        glVertexAttribDivisor(2, 1);
        // Attribute 3: Circle.starting_angle (float)
        glVertexAttribPointer(
            3, 1, GL_FLOAT, GL_FALSE,
            sizeof(Circle), (void*)offsetof(Circle, starting_angle)
        );
        glEnableVertexAttribArray(3);
        glVertexAttribDivisor(3, 1);
        // Attribute 4: Circle.frequency (float)
        glVertexAttribPointer(
            4, 1, GL_FLOAT, GL_FALSE,
            sizeof(Circle), (void*)offsetof(Circle, frequency)
        );
        glEnableVertexAttribArray(4);
        glVertexAttribDivisor(4, 1);
    }

    void updateInstanceBuffer()
    {
        // Upload your circles directly (NO CircleInstanceGPU, no copying)
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferData(GL_ARRAY_BUFFER,
                     circles.size() * sizeof(Circle),
                     circles.data(),
                     GL_DYNAMIC_DRAW);
    }

    void updateProjection(int w, int h)
    {
        float aspect = static_cast<float>(h) / static_cast<float>(w);

        glm::mat4 proj = glm::ortho(-1.0f, 1.0f,
                                    -aspect, aspect);

        myShader.use();
        GLint loc = glGetUniformLocation(myShader.ID, "projection");
        if (loc != -1)
        {
            glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(proj));
        }
    }

private:
    GLuint VAO = 0;
    GLuint meshVBO = 0;
    GLuint instanceVBO = 0;
    Shader myShader;

    int vertexCount = 0;
    int windowWidth = 1;
    int windowHeight = 1;

    std::vector<Circle> circles;

    static constexpr int SEGMENT_NUMBER = 120;
};

class PathRenderer
{
public:

    void init(int w, int h){
        // verts.reserve(points.size() * 3);

        // for (auto& p : points){
        //     verts.push_back(p.x);
        //     verts.push_back(p.y);
        //     verts.push_back(0.0f);
        // }

        // vertexCount = verts.size() / 3;
        windowHeight = w;
        windowWidth = w;

        pathShader.init(pathVertCodeString, pathFragCodeString);
        updateProjection(windowWidth, windowHeight);

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER,
                MAX_VERTS * sizeof(float),
                nullptr,
                GL_DYNAMIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                              3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
    }

    void addVertex(glm::vec2 p){
        if (verts.size() >= MAX_VERTS){
            return;
        }
        if (verts.size() >= 2000 * timePerCycle){
            verts.erase(verts.begin());
            verts.erase(verts.begin());
            verts.erase(verts.begin());
        }
        verts.push_back(p.x);
        verts.push_back(p.y);
        verts.push_back(0.0f);
        vertexCount = verts.size() / 3;
        upload();
    }

    vector<float> getPoints(){
        return verts;
    }
    
    void upload(){
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER,
                    0,
                    verts.size() * sizeof(float),
                    verts.data());
    }

    void draw(){
        if (vertexCount < 2) return;
        pathShader.use();
        glBindVertexArray(VAO);

        glLineWidth(1.0f);
        glDrawArrays(GL_LINE_STRIP, 0, vertexCount);
    }

    void onResize(int w, int h)
    {
        windowWidth  = (w > 0) ? w : 1;
        windowHeight = (h > 0) ? h : 1;
        updateProjection(windowWidth, windowHeight);
    }

private:
    void updateProjection(int w, int h){
        float aspect = static_cast<float>(h) / static_cast<float>(w);

        glm::mat4 proj = glm::ortho(-1.0f, 1.0f,
                                    -aspect, aspect);

        pathShader.use();
        GLint loc = glGetUniformLocation(pathShader.ID, "projection");
        if (loc != -1)
        {
            glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(proj));
        }
    }

private:
    GLuint VAO = 0;
    GLuint VBO = 0;
    int vertexCount = 0;
    int windowWidth = 1;
    int windowHeight = 1;
    vector<float> verts;

    Shader pathShader;

    static constexpr size_t MAX_VERTS = 10000 * 3;
};

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
}

bool init_window(GLFWwindow* &window)
{
    bool success = true;
    if (!glfwInit())
    {
        cout << "Failed to initialize GLFW" << endl;
        success = false;
    }
    else 
    {
        glfwWindowHint(GLFW_SAMPLES, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME, NULL, NULL);

        if (window == NULL)
        {
            cout << "Failed to open GLFW window" << endl;
            success = false;
        }
        else 
        {
            glfwMakeContextCurrent(window);
            if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
            {
                cout << "Failed to initialize GLAD" << endl;
                success = false;
            }
            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            glViewport(0, 0, w, h);

        }
    }


    return success;
}

void close_window(GLFWwindow* &window) 
{
    glfwDestroyWindow(window);
    window = NULL;
    glfwTerminate();
}

void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

/*std::vector<float> drawCircle( GLfloat x, GLfloat y, GLfloat z, GLfloat radius, GLint numberOfSides )
{
    float aspect = (float)WINDOW_HEIGHT / WINDOW_WIDTH;
    std::vector<float> circle;
    float twoPi = 2.0f * M_PI;
    int segments = 1 + numberOfSides;

    for (int i = 0; i <= segments; i++)
    {
        float angle = twoPi * i / segments;
        circle.push_back(radius * cos(angle) * aspect);
        circle.push_back(radius * sin(angle));
        circle.push_back(0.0f);
    }

    return circle;
}*/

void saveFrame(const std::string& filename, int w, int h){
    std::vector<unsigned char> pixels(w * h * 4);

    glReadPixels(
        0, 0, w, h,
        GL_RGBA, GL_UNSIGNED_BYTE,
        pixels.data()
    );

    // Flip vertically (OpenGL origin is bottom-left)
    for (int y = 0; y < h / 2; y++) {
        for (int x = 0; x < w * 4; x++) {
            std::swap(
                pixels[y * w * 4 + x],
                pixels[(h - 1 - y) * w * 4 + x]
            );
        }
    }

    stbi_write_png(
        filename.c_str(),
        w, h,
        4,
        pixels.data(),
        w * 4
    );
}

int main()
{
    GLFWwindow* window;

    if (!init_window(window)) {
        close_window(window);
        return -1;
    }
    
   
    CircleRenderer renderer;
    
    renderer.init(
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        vertexCodeString,
        fragmentCodeString
    );
    glfwSetWindowUserPointer(window, &renderer);

    PathRenderer path;
    path.init(WINDOW_WIDTH, WINDOW_HEIGHT);

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    renderer.onResize(w, h);
    path.onResize(w, h);


    std::vector<Circle> circles;

    std::vector<Point> points;
    points = processSVG(image);
    // printf("%i\n", points.size());
    // for (int x = 0; x<points.size(); x++){
    //     printf("%f, %f\n", points[x].x, points[x].y);
    // }

    points = removeDuplicates(points);

    points = normalize_points(points);

    points = resampleByArcLength(points);

    fftw_complex *output = fft_test(points.size(), points);

    // for (int x = 0; x<normalized_points.size(); x++){
    //     printf("%f, %f\n", normalized_points[x].x, normalized_points[x].y);
    // }

    circles.clear();
    circles.reserve(NUM_CIRCLES);

    for (int i = 0; i < NUM_CIRCLES; ++i)
    {
        circles.emplace_back(
            i,
            output,
            NUM_CIRCLES
        );
    }

    renderer.setCircles(circles);

    glfwSetFramebufferSizeCallback(
        window,
        [](GLFWwindow* win, int w, int h)
        {
            glViewport(0, 0, w, h);

            auto* renderer =
                static_cast<CircleRenderer*>(
                    glfwGetWindowUserPointer(win)
                );

            if (renderer)
                renderer->onResize(w, h);
        }
    );
    static int frame = 0;

    while(!glfwWindowShouldClose(window))
    {
        //input
        processInput(window);

        float time = (float)frame/fps;
        glm::vec2 pos(0.0f);

        for (auto& c : circles){
            float a = c.starting_angle + c.frequency * time;
            c.position = glm::vec3(pos, 0.0f);
            pos += c.radius * glm::vec2(cos(a), sin(a));
        }
        // outline.push_back(pos);

        renderer.setCircles(circles);

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        renderer.draw(time);
        path.addVertex(pos);
        path.draw();

        //check and call events and swap the buffers
        glfwSwapBuffers(window);
        glfwPollEvents();    

        if (recording)
        {
            char name[256];
            sprintf(name, "./frames/frame_%05d.png", frame++);
            saveFrame(name, WINDOW_WIDTH, WINDOW_HEIGHT);
        }
        if (frame > fps * timePerCycle * twoPi * 1.5)
            glfwSetWindowShouldClose(window, true);
    }

    close_window(window);
    glfwTerminate();

    return 0;
}

