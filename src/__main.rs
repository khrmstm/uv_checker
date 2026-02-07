use image::{GrayImage, ImageReader, Luma, Rgba, RgbaImage};
use imageproc::distance_transform::Norm;
use imageproc::morphology::dilate_mut;
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

  // 各ラベルごとにチェック
  let mut label_status: HashMap<u32, (bool, bool)> = HashMap::new();
  let labels: Vec<u32> = label_info.keys().copied().collect();

  let margin_threshold = 8u8; // now u8 for dilate_mut

  // One mask image per label, same size as UV
  let mut label_masks: HashMap<u32, GrayImage> = HashMap::new();
  for (x, y, pixel) in labeled.enumerate_pixels() {
    let label = pixel[0] as u32;
    if label == 0 {
      continue;
    }
    let mask = label_masks
      .entry(label)
      .or_insert_with(|| GrayImage::new(width, height));
    mask.put_pixel(x, y, Luma([255u8]));
  }

  // Dilate each island mask by margin_threshold
  let mut dilated_masks: HashMap<u32, GrayImage> = HashMap::new();
  for (label, mask) in &label_masks {
    let mut dilated = mask.clone();
    dilate_mut(&mut dilated, Norm::LInf, margin_threshold); // Chebyshev (square) dilation
    dilated_masks.insert(*label, dilated);
  }

  // Per‑label margin check: see if any two dilated masks overlap
  let mut label_status: HashMap<u32, (bool, bool)> = HashMap::new();
  let labels: Vec<u32> = label_info.keys().copied().collect();

  for &label in &labels {
    let ((min_x, min_y), (max_x, max_y), pixel_count) = label_info[&label];
    let bbox_area =
      (max_x.saturating_sub(min_x) + 1) as u64 * (max_y.saturating_sub(min_y) + 1) as u64;

    // Density / overlap/hole check (unchanged)
    let is_problem = pixel_count * 3 < bbox_area as u32;

    // Margin check: dilated label vs all others
    let dilated_self = &dilated_masks[&label];
    let mut margin_ok = true;

    for &other_label in &labels {
      if other_label == label {
        continue;
      }
      let dilated_other = &dilated_masks[&other_label];

      // Check for overlap in dilated masks
      let mut overlap = false;
      for y in 0..height {
        for x in 0..width {
          let v1 = dilated_self.get_pixel(x, y)[0];
          let v2 = dilated_other.get_pixel(x, y)[0];
          if v1 > 0 && v2 > 0 {
            overlap = true;
            break;
          }
        }
        if overlap {
          break;
        }
      }
      if overlap {
        margin_ok = false;
        break;
      }
    }

    label_status.insert(label, (is_problem, margin_ok));
  }

  // One big margin mask: all pixels within margin_threshold of any island
  let mut margin_mask = GrayImage::new(width, height);

  for dilated in dilated_masks.values() {
    for y in 0..height {
      for x in 0..width {
        let v = dilated.get_pixel(x, y)[0];
        if v > 0 {
          margin_mask.put_pixel(x, y, Luma([255u8]));
        }
      }
    }
  }

  // 結果画像生成

  // After you compute margin_mask, optionally:
  let mut conflict_mask = GrayImage::new(width, height);

  for &label in &labels {
    let dilated_self = &dilated_masks[&label];
    for &other_label in &labels {
      if other_label == label {
        continue;
      }
      let dilated_other = &dilated_masks[&other_label];
      for y in 0..height {
        for x in 0..width {
          let v1 = dilated_self.get_pixel(x, y)[0];
          let v2 = dilated_other.get_pixel(x, y)[0];
          if v1 > 0 && v2 > 0 {
            conflict_mask.put_pixel(x, y, Luma([255u8]));
          }
        }
      }
    }
  }

  // Then in the result_img loop:
  for (x, y, pixel) in labeled.enumerate_pixels() {
    let label = pixel[0] as u32;
    let margin = margin_mask.get_pixel(x, y)[0] > 0;
    let conflict = conflict_mask.get_pixel(x, y)[0] > 0;

    // Check if this pixel is on the image border
    let on_border = x == 0 || x == width - 1 || y == 0 || y == height - 1;
    let on_border_margin: bool = margin && on_border;

    let pixel = if on_border_margin {
      Rgba([128, 0, 128, 255]) // violet: touches border (U=0,1 or V=0,1)
    } else if label == 0 {
      if conflict {
        Rgba([255, 0, 0, 255]) // red: overlapping margin (no island)
      } else if margin {
        Rgba([0, 255, 255, 255]) // cyan: clean margin (no island)
      } else {
        Rgba([0, 0, 0, 0]) // transparent
      }
    } else {
      if conflict {
        Rgba([255, 0, 0, 255]) // red: island in conflict zone
      } else if margin {
        Rgba([0, 255, 255, 255]) // cyan: island in safe margin
      } else {
        Rgba([0, 255, 0, 255]) // green: island interior (no margin)
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
