use image::{GrayImage, ImageReader, Luma, Rgba, RgbaImage};
use imageproc::region_labelling::{Connectivity, connected_components};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let img = ImageReader::open("uv_layout.png")?.decode()?.to_rgba8();
  let (width, height) = img.dimensions();
  let mut img_grey = GrayImage::new(width, height);

  // 透明ピクセル以外を前景(255)に変換
  for (x, y, pixel) in img.enumerate_pixels() {
    let alpha = pixel[3];
    img_grey.put_pixel(x, y, Luma([if alpha > 128 { 255u8 } else { 0u8 }]));
  }

  // 連結成分ラベリング（デフォルト8近傍）
  let background = Luma([0u8]);
  let labeled = connected_components(&img_grey, Connectivity::Eight, background);

  // 各ラベル領域の境界ボックスを計算 + ピクセルカウント
  let mut label_info: HashMap<u32, ((u32, u32), (u32, u32), u32)> = HashMap::new();
  // bounds + pixel_count

  for (x, y, pixel) in labeled.enumerate_pixels() {
    let label = pixel[0] as u32;
    if label == 0 {
      continue;
    }

    let entry = label_info
      .entry(label)
      .or_insert(((u32::MAX, u32::MAX), (0, 0), 0));
    entry.0.0 = entry.0.0.min(x);
    entry.0.1 = entry.0.1.min(y);
    entry.1.0 = entry.1.0.max(x);
    entry.1.1 = entry.1.1.max(y);
    entry.2 += 1;
  }

  let mut result_img = RgbaImage::new(width, height);
  let margin_threshold = 32u32;

  // 各ラベルごとにチェック
  let mut label_status: HashMap<u32, (bool, bool)> = HashMap::new();
  let labels: Vec<u32> = label_info.keys().copied().collect();

  for &label in &labels {
    let ((min_x, min_y), (max_x, max_y), pixel_count) = label_info[&label];
    let bbox_area =
      ((max_x.saturating_sub(min_x) + 1) as u64 * (max_y.saturating_sub(min_y) + 1) as u64) as u32;

    // 重なり/穴チェック: ピクセル密度低
    let is_problem = pixel_count * 3 < bbox_area;

    // マージンチェック
    let mut margin_ok = true;
    for &other_label in &labels {
      if other_label == label {
        continue;
      }
      let ((oxmin, oymin), (oxmax, oymax), _) = label_info[&other_label];
      let dist = min_distance_bounds(
        ((min_x, min_y), (max_x, max_y)),
        ((oxmin, oymin), (oxmax, oymax)),
      );
      if dist < margin_threshold {
        margin_ok = false;
        break;
      }
    }

    label_status.insert(label, (is_problem, margin_ok));
  }

  // 結果画像生成
  for (x, y, pixel) in labeled.enumerate_pixels() {
    let label = pixel[0] as u32;
    let pixel = if label == 0 {
      Rgba([0, 0, 0, 0])
    } else {
      let (problem, margin_ok) = label_status[&label];
      match (problem, margin_ok) {
        (true, _) => Rgba([255, 0, 0, 255]),        // 赤: 密度問題
        (false, false) => Rgba([255, 255, 0, 255]), // 黄: マージンNG
        (false, true) => Rgba([0, 255, 0, 255]),    // 緑: OK
      }
    };
    result_img.put_pixel(x, y, pixel);
  }

  result_img.save("uv_analysis.png")?;

  // 統計出力
  println!("全ラベル数: {}", labels.len());
  println!("問題ラベル:");
  let mut issues = 0;
  for &label in &labels {
    let ((min_x, min_y), (max_x, max_y), pixel_count) = label_info[&label];
    let (problem, margin_ok) = label_status[&label];
    if problem || !margin_ok {
      let bbox_area =
        (max_x.saturating_sub(min_x) + 1) as u32 * (max_y.saturating_sub(min_y) + 1) as u32;
      let density = if bbox_area > 0 {
        (pixel_count as f32 / bbox_area as f32 * 100.0) as u32
      } else {
        0
      };
      println!(
        "  L{}: pix={} ({}%), size={}x{}, margin={}",
        label,
        pixel_count,
        density,
        max_x.saturating_sub(min_x) + 1,
        max_y.saturating_sub(min_y) + 1,
        margin_ok
      );
      issues += 1;
    }
  }
  println!("問題数: {}/{}", issues, labels.len());

  Ok(())
}

/// 2つの境界ボックスの最小距離
fn min_distance_bounds(b1: ((u32, u32), (u32, u32)), b2: ((u32, u32), (u32, u32))) -> u32 {
  let ((x1min, y1min), (x1max, y1max)) = b1;
  let ((x2min, y2min), (x2max, y2max)) = b2;

  let dx = if x1max < x2min {
    x2min.saturating_sub(x1max)
  } else if x2max < x1min {
    x1min.saturating_sub(x2max)
  } else {
    0
  };

  let dy = if y1max < y2min {
    y2min.saturating_sub(y1max)
  } else if y2max < y1min {
    y1min.saturating_sub(y2max)
  } else {
    0
  };

  dx.min(dy)
}
