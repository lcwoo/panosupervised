# Original implementation: https://github.com/xuxy09/QVI/blob/master/models/forward_warp_gaussian.py
import torch
import torch.nn as nn


class FlowReversal(nn.Module):
	def __init__(self):
		super(FlowReversal, self).__init__()

	def forward(self, src_img, src_flo, normalized_flo=True, dst_img_shape=None):
		"""
			# (x1, y1)		(x1, y2)
			# +---------------+
			# |				  |
			# |	o(x, y) 	  |
			# |				  |
			# |				  |
			# |				  |
			# |				  |
			# +---------------+
			# (x2, y1)		(x2, y2)

			-img: image (B, C, H, W)
			-flo: optical flow (B, H, W, 2)
			elements of flo is in [0, H] and [0, W] for dx, dy
		"""
		B, C = src_img.shape[:2]
		src_img_shape = src_img.shape
		dst_img_shape = (B, C, *dst_img_shape) if dst_img_shape is not None else src_img.shape

		Hd, Wd = dst_img_shape[2:]

		assert normalized_flo == True

		# translate start-point optical flow to end-point optical flow
		x = src_flo[..., 0]
		y = src_flo[..., 1]

		x = (x + 1) / 2 * Wd
		y = (y + 1) / 2 * Hd

		x = x[:, None].repeat(1, C, 1, 1)
		y = y[:, None].repeat(1, C, 1, 1)

        # TODO(soonminh): Clean up this implementation, improve efficiency

		# # Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
		# x1 = torch.floor(x)
		# x2 = x1 + 1
		# y1 = torch.floor(y)
		# y2 = y1 + 1

		# # firstly, get gaussian weights
		# w11, w12, w21, w22 = self.get_gaussian_weights(x, y, x1, x2, y1, y2)

		# # secondly, sample each weighted corner
		# img11, o11 = self.sample_one(src_img, x1, y1, w11, src_img_shape, dst_img_shape)
		# img12, o12 = self.sample_one(src_img, x1, y2, w12, src_img_shape, dst_img_shape)
		# img21, o21 = self.sample_one(src_img, x2, y1, w21, src_img_shape, dst_img_shape)
		# img22, o22 = self.sample_one(src_img, x2, y2, w22, src_img_shape, dst_img_shape)

		# imgw = img11 + img12 + img21 + img22
		# o = o11 + o12 + o21 + o22

		# # 8 points of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
		# x0 = torch.floor(x) - 1
		# x1 = torch.floor(x)
		# x2 = x1 + 1
		# x3 = x1 + 2

		# y0 = torch.floor(y) - 1
		# y1 = torch.floor(y)
		# y2 = y1 + 1
		# y3 = y1 + 2

		# # firstly, get gaussian weights
		# # w11, w12, w21, w22 = self.get_gaussian_weights(x, y, x1, x2, y1, y2)
		# w00 = self.get_gaussian_weights(x, y, x0, y0)
		# # w01 = self.get_gaussian_weights(x, y, x0, y1)
		# # w02 = self.get_gaussian_weights(x, y, x0, y2)
		# w03 = self.get_gaussian_weights(x, y, x0, y3)

		# # w10 = self.get_gaussian_weights(x, y, x1, y0)
		# w11 = self.get_gaussian_weights(x, y, x1, y1)
		# w12 = self.get_gaussian_weights(x, y, x1, y2)
		# # w13 = self.get_gaussian_weights(x, y, x1, y3)

		# # w20 = self.get_gaussian_weights(x, y, x2, y0)
		# w21 = self.get_gaussian_weights(x, y, x2, y1)
		# w22 = self.get_gaussian_weights(x, y, x2, y2)
		# # w23 = self.get_gaussian_weights(x, y, x2, y3)

		# w30 = self.get_gaussian_weights(x, y, x3, y0)
		# # w31 = self.get_gaussian_weights(x, y, x3, y1)
		# # w32 = self.get_gaussian_weights(x, y, x3, y2)
		# w33 = self.get_gaussian_weights(x, y, x3, y3)


		# # secondly, sample each weighted corner
		# img00, o00 = self.sample_one(src_img, x0, y0, w00, src_img_shape, dst_img_shape)
		# # img01, o01 = self.sample_one(src_img, x0, y1, w01, src_img_shape, dst_img_shape)
		# # img02, o02 = self.sample_one(src_img, x0, y2, w02, src_img_shape, dst_img_shape)
		# img03, o03 = self.sample_one(src_img, x0, y3, w03, src_img_shape, dst_img_shape)

		# # img10, o10 = self.sample_one(src_img, x1, y0, w10, src_img_shape, dst_img_shape)
		# img11, o11 = self.sample_one(src_img, x1, y1, w11, src_img_shape, dst_img_shape)
		# img12, o12 = self.sample_one(src_img, x1, y2, w12, src_img_shape, dst_img_shape)
		# # img13, o13 = self.sample_one(src_img, x1, y3, w13, src_img_shape, dst_img_shape)

		# # img20, o20 = self.sample_one(src_img, x2, y0, w20, src_img_shape, dst_img_shape)
		# img21, o21 = self.sample_one(src_img, x2, y1, w21, src_img_shape, dst_img_shape)
		# img22, o22 = self.sample_one(src_img, x2, y2, w22, src_img_shape, dst_img_shape)
		# # img23, o23 = self.sample_one(src_img, x2, y3, w23, src_img_shape, dst_img_shape)

		# img30, o30 = self.sample_one(src_img, x3, y0, w30, src_img_shape, dst_img_shape)
		# # img31, o31 = self.sample_one(src_img, x3, y1, w31, src_img_shape, dst_img_shape)
		# # img32, o32 = self.sample_one(src_img, x3, y2, w32, src_img_shape, dst_img_shape)
		# img33, o33 = self.sample_one(src_img, x3, y3, w33, src_img_shape, dst_img_shape)

		# imgw = img00 + img03 \
		# 	 + img11 + img12 \
		# 	 + img21 + img22 \
		# 	 + img30 + img33

		# o = o00 + o03 \
		#   + o11 + o12 \
		#   + o21 + o22 \
		#   + o30 + o33

		# 16 points of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
		x0 = torch.floor(x) - 1
		x1 = torch.floor(x)
		x2 = x1 + 1
		x3 = x1 + 2

		y0 = torch.floor(y) - 1
		y1 = torch.floor(y)
		y2 = y1 + 1
		y3 = y1 + 2

		# firstly, get gaussian weights
		# w11, w12, w21, w22 = self.get_gaussian_weights(x, y, x1, x2, y1, y2)
		w00 = self.get_gaussian_weights(x, y, x0, y0)
		w01 = self.get_gaussian_weights(x, y, x0, y1)
		w02 = self.get_gaussian_weights(x, y, x0, y2)
		w03 = self.get_gaussian_weights(x, y, x0, y3)

		w10 = self.get_gaussian_weights(x, y, x1, y0)
		w11 = self.get_gaussian_weights(x, y, x1, y1)
		w12 = self.get_gaussian_weights(x, y, x1, y2)
		w13 = self.get_gaussian_weights(x, y, x1, y3)

		w20 = self.get_gaussian_weights(x, y, x2, y0)
		w21 = self.get_gaussian_weights(x, y, x2, y1)
		w22 = self.get_gaussian_weights(x, y, x2, y2)
		w23 = self.get_gaussian_weights(x, y, x2, y3)

		w30 = self.get_gaussian_weights(x, y, x3, y0)
		w31 = self.get_gaussian_weights(x, y, x3, y1)
		w32 = self.get_gaussian_weights(x, y, x3, y2)
		w33 = self.get_gaussian_weights(x, y, x3, y3)


		# secondly, sample each weighted corner
		img00, o00 = self.sample_one(src_img, x0, y0, w00, src_img_shape, dst_img_shape)
		img01, o01 = self.sample_one(src_img, x0, y1, w01, src_img_shape, dst_img_shape)
		img02, o02 = self.sample_one(src_img, x0, y2, w02, src_img_shape, dst_img_shape)
		img03, o03 = self.sample_one(src_img, x0, y3, w03, src_img_shape, dst_img_shape)

		img10, o10 = self.sample_one(src_img, x1, y0, w10, src_img_shape, dst_img_shape)
		img11, o11 = self.sample_one(src_img, x1, y1, w11, src_img_shape, dst_img_shape)
		img12, o12 = self.sample_one(src_img, x1, y2, w12, src_img_shape, dst_img_shape)
		img13, o13 = self.sample_one(src_img, x1, y3, w13, src_img_shape, dst_img_shape)

		img20, o20 = self.sample_one(src_img, x2, y0, w20, src_img_shape, dst_img_shape)
		img21, o21 = self.sample_one(src_img, x2, y1, w21, src_img_shape, dst_img_shape)
		img22, o22 = self.sample_one(src_img, x2, y2, w22, src_img_shape, dst_img_shape)
		img23, o23 = self.sample_one(src_img, x2, y3, w23, src_img_shape, dst_img_shape)

		img30, o30 = self.sample_one(src_img, x3, y0, w30, src_img_shape, dst_img_shape)
		img31, o31 = self.sample_one(src_img, x3, y1, w31, src_img_shape, dst_img_shape)
		img32, o32 = self.sample_one(src_img, x3, y2, w32, src_img_shape, dst_img_shape)
		img33, o33 = self.sample_one(src_img, x3, y3, w33, src_img_shape, dst_img_shape)

		imgw = img00 + img01 + img02 + img03 \
			 + img10 + img11 + img12 + img13 \
			 + img20 + img21 + img22 + img23 \
			 + img30 + img31 + img32 + img33

		o = o00 + o01 + o02 + o03 \
		  + o10 + o11 + o12 + o13 \
		  + o20 + o21 + o22 + o23 \
		  + o30 + o31 + o32 + o33

		return imgw, o

	def get_gaussian_weights(self, x, y, x1, y1):
		return torch.exp(-((x - x1)**2 + (y - y1)**2))


	# def get_gaussian_weights(self, x, y, x1, x2, y1, y2):
	# 	w11 = torch.exp(-((x - x1)**2 + (y - y1)**2))
	# 	w12 = torch.exp(-((x - x1)**2 + (y - y2)**2))
	# 	w21 = torch.exp(-((x - x2)**2 + (y - y1)**2))
	# 	w22 = torch.exp(-((x - x2)**2 + (y - y2)**2))
	# 	return w11, w12, w21, w22


	def sample_one(self, img, x, y, weight, src_img_shape, dst_img_shape):
		"""
		Input:
			-img (N, C, H, W)
			-shiftx, shifty (N, c, H, W)
		"""
		device = img.device
		Ns, Cs, Hs, Ws = src_img_shape
		Nd, Cd, Hd, Wd = dst_img_shape

		# Prepare index tensor from source domain (e.g. PanoSpace)
		idxn = torch.arange(0, Ns, device=device, requires_grad=False)
		idxc = torch.arange(0, Cs, device=device, requires_grad=False)
		idxy = torch.arange(0, Hs, device=device, requires_grad=False)
		idxx = torch.arange(0, Ws, device=device, requires_grad=False)

		idxn, idxc, idxy, idxx = torch.meshgrid([idxn, idxc, idxy, idxx], indexing='ij')

		flat_idxn = idxn.contiguous().view(-1)
		flat_idxc = idxc.contiguous().view(-1)
		# flat_idxx = idxx.contiguous().view(-1)
		# flat_idxy = idxy.contiguous().view(-1)

		flat_weight = weight.view(-1)
		flat_img = img.view(-1)

		# Prepare value tensor, i.e. index of destination domain, from source domain
		valn = flat_idxn.long()
		valc = flat_idxc.long()
		valx = x.view(-1).long()
		valy = y.view(-1).long()

		# Mask out out-of-boundary in destination domain
		mask = valx.ge(0) & valx.lt(Wd) & valy.ge(0) & valy.lt(Hd)

		# Mask off points out of boundaries
		valn_masked = torch.masked_select(valn, mask).clone().to(device)
		valc_masked = torch.masked_select(valc, mask).clone().to(device)
		valx_masked = torch.masked_select(valx, mask).clone().to(device)
		valy_masked = torch.masked_select(valy, mask).clone().to(device)


		# TODO(soonminh): consider index_put_ order by depth in camera frame, to have z-buffering effect
		# mask_select -> put
		# Note here! accmulate fla must be true for proper bp
		img_warp = torch.zeros([Nd, Cd, Hd, Wd], device=device)
		img_warp.index_put_((valn_masked, valc_masked, valy_masked, valx_masked), torch.masked_select(flat_img * flat_weight, mask), accumulate=True)

		one_warp = torch.zeros([Nd, Cd, Hd, Wd], device=device)
		one_warp.index_put_((valn_masked, valc_masked, valy_masked, valx_masked), torch.masked_select(flat_weight, mask), accumulate=True)

		return img_warp, one_warp
