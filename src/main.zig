// raylib-zig (c) Nikolas Wipper 2023
const std = @import("std");
const expect = std.testing.expect;

const rl = @import("raylib");

const Fluid = struct {
    allocator: *std.mem.Allocator,
    size: i32,
    dt: f32,
    diff: f32, // diffusion
    visc: f32, // viscosity

    s: []f32,
    density: []f32,

    Vx: []f32,
    Vy: []f32,

    Vx0: []f32,
    Vy0: []f32,

    pub fn init(allocator: *std.mem.Allocator, size: i32, dt: f32, diff: f32, visc: f32) !Fluid {
        const usize_size = @as(usize, @intCast(size));
        const area = usize_size * usize_size;
        return Fluid{
            .allocator = allocator,
            .size = size,
            .dt = dt,
            .diff = diff,
            .visc = visc,

            .s = try allocator.alloc(f32, area),
            .density = try allocator.alloc(f32, area),

            .Vx = try allocator.alloc(f32, area),
            .Vy = try allocator.alloc(f32, area),

            .Vx0 = try allocator.alloc(f32, area),
            .Vy0 = try allocator.alloc(f32, area),
        };
    }
    pub fn deinit(self: *Fluid) void {
        //free memory
        self.allocator.free(self.s);
        self.allocator.free(self.density);
        self.allocator.free(self.Vx);
        self.allocator.free(self.Vy);
        self.allocator.free(self.Vx0);
        self.allocator.free(self.Vy0);
    }
    pub fn clampDensity(self: *Fluid) void {
        var i: usize = 0;
        while (i < self.size * self.size) : (i += 1) {
            self.density[i] = std.math.clamp(self.density[i], 0.0, 255.0);
        }
    }

    pub fn addDensity(self: *Fluid, x: i32, y: i32, amount: f32) !void {
        const N = self.size;
        const index = try IX(N, x, y);
        self.density[index] += amount;
    }

    pub fn addVelocity(self: *Fluid, x: i32, y: i32, amount_x: f32, amount_y: f32) !void {
        const N = self.size;
        const index = try IX(N, x, y);
        self.Vx[index] += amount_x;
        self.Vy[index] += amount_y;
    }

    pub fn fluid_step(self: *Fluid, iter: i32, N: i32) !void {
        try diffuse(1, self.Vx0, self.Vx, self.visc, self.dt, iter, N);
        try diffuse(2, self.Vy0, self.Vy, self.visc, self.dt, iter, N);

        try project(self.Vx0, self.Vy0, self.Vx, self.Vy, iter, N);

        try advect(1, self.Vx, self.Vx0, self.Vx0, self.Vy0, self.dt, N);
        try advect(2, self.Vy, self.Vy0, self.Vx0, self.Vy0, self.dt, N);

        try project(self.Vx, self.Vy, self.Vx0, self.Vy0, iter, N);

        try diffuse(0, self.s, self.density, self.diff, self.dt, iter, N);
        try advect(0, self.density, self.s, self.Vx, self.Vy, self.dt, N);
    }

    fn mouseDragged(self: *Fluid, scale: i32) !void {
        const x = rl.getMouseX();
        const y = rl.getMouseY();
        const x_scale = @divTrunc(x, scale);
        const y_scale = @divTrunc(y, scale);
        try self.addDensity(x_scale, y_scale, @as(f32, @floatFromInt(rl.getRandomValue(1, 255))));
        const mouseD = rl.getMouseDelta();
        const amount_x = mouseD.x * 10;
        const amount_y = mouseD.y * 10;
        try self.addVelocity(x_scale, y_scale, amount_x, amount_y);
    }

    fn render(self: *Fluid, scale: i32) !void {
        const size_usize = @as(usize, @intCast(self.size));
        for (0..size_usize) |j_usize| {
            const j = @as(i32, @intCast(j_usize));
            for (0..size_usize) |i_usize| {
                const i = @as(i32, @intCast(i_usize));
                const x = i * scale;
                const y = j * scale;
                const dens = self.density[try IX(self.size, i, j)];

                const dens_c = constrain_f(dens, 0, 255);
                const dens_u8 = @as(u8, @intFromFloat(dens_c));
                const color: rl.Color = .{
                    .r = dens_u8,
                    .g = dens_u8,
                    .b = dens_u8,
                    .a = 255,
                };
                rl.drawRectangle(x, y, scale, scale, color);
            }
        }
    }
};

pub fn main() anyerror!void {
    const N = 64;
    const scale = 14;
    const iterations = 16;
    // Initialization
    //--------------------------------------------------------------------------------------
    //const screenWidth = 800;
    //const screenHeight = 450;

    rl.initWindow(N * scale, N * scale, "raylib-zig [core] example - basic window");
    defer rl.closeWindow(); // Close window and OpenGL context

    rl.setTargetFPS(60); // Set our game to run at 60 frames-per-second
    //--------------------------------------------------------------------------------------
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var allocator = gpa.allocator();

    var fluid: Fluid = Fluid.init(&allocator, N, 1.0, 0, 0) catch |err| {
        std.log.err("Failed to initialize fluid: {}", .{err});
        return err;
    };

    // Main game loop
    while (!rl.windowShouldClose()) { // Detect window close button or ESC key
        // Update
        //----------------------------------------------------------------------------------
        // TODO: Update your variables here
        //----------------------------------------------------------------------------------

        // TODO: Add mouse dragging functionality
        try fluid.mouseDragged(scale);
        try fluid.fluid_step(iterations, N);
        fluid.clampDensity();

        // Draw
        //----------------------------------------------------------------------------------
        rl.beginDrawing();
        try fluid.render(scale);
        defer rl.endDrawing();

        rl.clearBackground(rl.Color.black);
        //----------------------------------------------------------------------------------
    }
    defer {
        const deinit_status = gpa.deinit();
        if (deinit_status == .leak) expect(false) catch @panic("Test failed");
    }
    defer fluid.deinit();
}

inline fn IX(N: i32, x: i32, y: i32) !usize {
    const x_c = constrain(x, 0, N - 1);
    const y_c = constrain(y, 0, N - 1);
    const ix_i = x_c + y_c * N;
    const ix_usize = try castIndex(ix_i);
    return ix_usize;
}

fn diffuse(b: i32, x: []f32, x0: []f32, diff: f32, dt: f32, iter: i32, N: i32) !void {
    const n_minus_two = @as(f32, @floatFromInt(N - 2));
    const a = dt * diff * n_minus_two * n_minus_two;
    try linear_solve(b, x, x0, a, 1 + 6 * a, iter, N);
}

fn linear_solve(b: i32, x: []f32, x0: []f32, a: f32, c: f32, iter: i32, N: i32) !void {
    const cRecip = 1.0 / c;
    const iter_usize = @as(usize, @intCast(iter));
    const N_usize = @as(usize, @intCast(N));
    for (0..iter_usize) |_| {
        for (0..N_usize - 1) |j_usize| {
            const j = @as(i32, @intCast(j_usize));
            for (0..N_usize - 1) |i_usize| {
                const i = @as(i32, @intCast(i_usize));
                x[try IX(N, i, j)] = (x0[try IX(N, i, j)] + a * (x[try IX(N, i + 1, j)] + x[try IX(N, i - 1, j)] + x[try IX(N, i, j + 1)] + x[try IX(N, i, j - 1)])) * cRecip;
            }
        }
        try set_bnd(b, x, N);
    }
}

fn project(velocX: []f32, velocY: []f32, p: []f32, div: []f32, iter: i32, N: i32) !void {
    const N_usize = @as(usize, @intCast(N));
    const N_f = @as(f32, @floatFromInt(N));
    for (0..N_usize - 1) |j_usize| {
        const j = @as(i32, @intCast(j_usize));
        for (0..N_usize - 1) |i_usize| {
            const i = @as(i32, @intCast(i_usize));
            div[try IX(N, i, j)] = -0.5 * (velocX[try IX(N, i + 1, j)] - velocX[try IX(N, i - 1, j)] + velocY[try IX(N, i, j + 1)] - velocY[try IX(N, i, j - 1)]) / N_f;
        }
    }
    try set_bnd(0, div, N);
    try set_bnd(0, p, N);
    try linear_solve(0, p, div, 1, 6, iter, N);

    for (0..N_usize - 1) |j_usize| {
        const j = @as(i32, @intCast(j_usize));
        for (0..N_usize - 1) |i_usize| {
            const i = @as(i32, @intCast(i_usize));
            velocX[try IX(N, i, j)] -= 0.5 * (p[try IX(N, i + 1, j)] - p[try IX(N, i - 1, j)]) / N_f;
            velocY[try IX(N, i, j)] -= 0.5 * (p[try IX(N, i, j + 1)] - p[try IX(N, i, j - 1)]) / N_f;
        }
    }
    try set_bnd(1, velocX, N);
    try set_bnd(2, velocY, N);
}

fn advect(b: i32, d: []f32, d0: []f32, velocX: []f32, velocY: []f32, dt: f32, N: i32) !void {
    var k0: f32 = undefined;
    var k1: f32 = undefined;
    var j0: f32 = undefined;
    var j1: f32 = undefined;

    const n_minus_two_f = @as(f32, @floatFromInt(N - 2));
    const dtx = dt * n_minus_two_f;
    const dty = dt * n_minus_two_f;

    var s0: f32 = undefined;
    var s1: f32 = undefined;
    var t0: f32 = undefined;
    var t1: f32 = undefined;

    var tmp1: f32 = undefined;
    var tmp2: f32 = undefined;

    var x: f32 = undefined;
    var y: f32 = undefined;

    const Nf = @as(f32, @floatFromInt(N));
    const N_usize = @as(usize, @intCast(N));
    for (0..N_usize - 1) |j_usize| {
        const j = @as(i32, @intCast(j_usize));
        for (0..N_usize - 1) |k_usize| {
            const k = @as(i32, @intCast(k_usize));
            tmp1 = dtx * velocX[try IX(N, k, j)];
            tmp2 = dty * velocY[try IX(N, k, j)];
            x = @as(f32, @floatFromInt(k)) - tmp1;
            y = @as(f32, @floatFromInt(j)) - tmp2;

            if (x < 0.5) x = 0.5;
            if (x > Nf - 0.5) x = Nf - 0.5;
            k0 = @floor(x);
            k1 = k0 + 1.0;

            if (y < 0.5) y = 0.5;
            if (y > Nf - 0.5) y = Nf - 0.5;
            j0 = @floor(y);
            j1 = j0 + 1.0;

            s1 = x - k0;
            s0 = 1.0 - s1;
            t1 = y - j0;
            t0 = 1.0 - t1;

            const k0i = @as(i32, @intFromFloat(k0));
            const k1i = @as(i32, @intFromFloat(k1));
            const j0i = @as(i32, @intFromFloat(j0));
            const j1i = @as(i32, @intFromFloat(j1));

            d[try IX(N, k, j)] = s0 * (t0 * d0[try IX(N, k0i, j0i)] + t1 * d0[try IX(N, k0i, j1i)]) +
                s1 * (t0 * d0[try IX(N, k1i, j0i)] + t1 * d0[try IX(N, k1i, j1i)]);
        }
    }
    try set_bnd(b, d, N);
}

fn set_bnd(b: i32, x: []f32, N: i32) !void {
    const N_usize = @as(usize, @intCast(N));
    for (0..N_usize - 1) |i_usize| {
        const i = @as(i32, @intCast(i_usize));
        x[try IX(N, i, 0)] = if (b == 2) -x[try IX(N, i, 1)] else x[try IX(N, i, 1)];
        x[try IX(N, i, N - 1)] = if (b == 2) -x[try IX(N, i, N - 2)] else x[try IX(N, i, N - 2)];
    }

    for (0..N_usize - 1) |i_usize| {
        const i = @as(i32, @intCast(i_usize));
        x[try IX(N, 0, i)] = if (b == 1) -x[try IX(N, 1, i)] else x[try IX(N, 1, i)];
        x[try IX(N, N - 1, i)] = if (b == 1) -x[try IX(N, N - 2, i)] else x[try IX(N, N - 2, i)];
    }

    x[try IX(N, 0, 0)] = 0.5 * (x[try IX(N, 1, 0)] + x[try IX(N, 0, 1)]);
    x[try IX(N, 0, N - 1)] = 0.5 * (x[try IX(N, 1, N - 1)] + x[try IX(N, 0, N - 2)]);
    x[try IX(N, N - 1, 0)] = 0.5 * (x[try IX(N, N - 2, 0)] + x[try IX(N, N - 1, 1)]);
    x[try IX(N, N - 1, N - 1)] = 0.5 * (x[try IX(N, N - 2, N - 1)] + x[try IX(N, N - 1, N - 2)]);
}

fn castIndex(index: i32) !usize {
    if (index < 0) return error.Overflow;
    if (index > std.math.maxInt(usize)) return error.Overflow;
    return @as(usize, @intCast(index));
}
// constrain() Description
//Constrains a value to not exceed a maximum and minimum value.
fn constrain(x: i32, min: i32, max: i32) i32 {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

fn constrain_f(x: f32, min: f32, max: f32) f32 {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}
