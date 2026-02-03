use criterion::{criterion_group, criterion_main, Criterion, black_box};
use foldit_render::easing::EasingFunction;
use foldit_render::animation::{ResidueAnimationState, AnimationTimeline};
use glam::Vec3;
use std::time::Duration;

fn easing_benchmark(c: &mut Criterion) {
    let f = EasingFunction::CubicHermite { c1: 0.33, c2: 1.0 };
    c.bench_function("cubic_hermite_easing", |b| {
        b.iter(|| black_box(f.evaluate(black_box(0.5))))
    });
}

fn residue_interpolation_benchmark(c: &mut Criterion) {
    let state = ResidueAnimationState::new(
        0,
        [Vec3::ZERO, Vec3::X, Vec3::Y],
        [Vec3::ONE, Vec3::new(2.0, 1.0, 0.0), Vec3::Z],
        &[0.0, 45.0, 90.0, 135.0],
        &[90.0, 135.0, 180.0, 225.0],
    );

    c.bench_function("single_residue_interpolation", |b| {
        b.iter(|| black_box(state.interpolate(black_box(0.5))))
    });
}

fn timeline_update_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("timeline_update");

    for count in [10, 50, 100, 500].iter() {
        let mut timeline = AnimationTimeline::new(*count);
        let states: Vec<ResidueAnimationState> = (0..*count)
            .map(|i| ResidueAnimationState::new(
                i,
                [Vec3::ZERO; 3],
                [Vec3::ONE; 3],
                &[],
                &[],
            ))
            .collect();
        timeline.add(states, Some(Duration::from_millis(300)), None);

        group.bench_function(format!("{}_residues", count), |b| {
            b.iter(|| {
                black_box(timeline.update(std::time::Instant::now()))
            })
        });
    }
    group.finish();
}

criterion_group!(benches, easing_benchmark, residue_interpolation_benchmark, timeline_update_benchmark);
criterion_main!(benches);
