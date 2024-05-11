const Fluid = struct {
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

    pub fn init(size: i32, dt: f32, diff: f32, visc: f32) Fluid {
        return Fluid{
            .size = size,
            .dt = dt,
            .diff = diff,
            .visc = visc,

            .s = [_]f32{0} ** (size * size),
            .density = [_]f32{0} ** (size * size),

            .Vx = [_]f32{0} ** (size * size),
            .Vy = [_]f32{0} ** (size * size),

            .Vx0 = [_]f32{0} ** (size * size),
            .Vy0 = [_]f32{0} ** (size * size),
        };
    }
    pub fn deinit(self: *Fluid) void {
        //free memory
        
    }
};